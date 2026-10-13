"""Scheduled ValidMind validation pipeline for OpenShift AI (Kubeflow Pipelines v2).

Same guts as the quickstart notebook — load data (S3 connection or synthetic
fallback), train the reference classifier, run the ValidMind tests, and send
results to the ValidMind Platform — packaged as a pipeline so it can run on a
schedule or on demand.

Compile:  python validmind_pipeline.py   →  dist/validmind_pipeline.yaml

The ValidMind credentials come from the quickstart's connection Secret,
attached to the task as environment variables at compile time via
kfp-kubernetes. Set VALIDMIND_SECRET_NAME (and optionally S3_SECRET_NAME for
an S3 data connection) before compiling to point at different secrets.
"""

import os

from kfp import compiler, dsl, kubernetes

VALIDMIND_SECRET_NAME = os.environ.get(
    "VALIDMIND_SECRET_NAME", "quickstart-validmind-connection"
)
S3_SECRET_NAME = os.environ.get("S3_SECRET_NAME", "")

VM_ENV_KEYS = ["VM_API_HOST", "VM_API_KEY", "VM_API_SECRET", "VM_API_MODEL"]
S3_ENV_KEYS = [
    "AWS_S3_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_BUCKET",
    "AWS_DEFAULT_REGION",
]


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-312:latest",
    # Same tested combination as the quickstart notebook: plotly<6 +
    # kaleido 0.2.1 (self-contained figure export — kaleido>=1 needs a Chrome
    # install the container doesn't have). jinja2 is imported by the validmind
    # library but not declared as its dependency; Jupyter images mask that,
    # a clean container doesn't.
    packages_to_install=["validmind", "plotly<6", "kaleido==0.2.1", "jinja2", "boto3"],
)
def run_validmind_validation(
    data_object_key: str,
    target_column: str,
    document: str,
    monitoring: bool,
) -> str:
    """Train the reference model and run the ValidMind test suite — the same
    steps as the quickstart notebook, headless."""
    import io
    import os

    import pandas as pd

    # ------------------------------------------------------------------ init
    # Workaround for a validmind library issue in plain (non-Jupyter) runs:
    # its API client caches an aiohttp session across calls, but each call
    # spins up a fresh asyncio.run() loop, so the second API call dies with
    # "RuntimeError: Event loop is closed". Keep one loop for the whole
    # process and route asyncio.run through it.
    import asyncio

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    asyncio.run = lambda coro, **kwargs: _loop.run_until_complete(coro)

    import validmind as vm

    # `document` names a specific document on the platform (e.g. for ongoing
    # monitoring); leaving it empty uses the model's default documentation,
    # exactly like the quickstart notebook's vm.init().
    init_kwargs = {
        "monitoring": monitoring,
        "generate_descriptions": False,  # headless: skip LLM description calls
    }
    if document:
        init_kwargs["document"] = document
    vm.init(**init_kwargs)
    vm.print_env()

    # ------------------------------------------------------------------ data
    # Use the S3 connection when its variables are present, otherwise the
    # same synthetic dataset as the notebook.
    if os.environ.get("AWS_S3_ENDPOINT"):
        import boto3

        session = boto3.session.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        s3 = session.client(
            "s3",
            endpoint_url=os.environ["AWS_S3_ENDPOINT"],
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
        )
        obj = s3.get_object(Bucket=os.environ["AWS_S3_BUCKET"], Key=data_object_key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        print(f"Loaded {len(df):,} rows from s3://{os.environ['AWS_S3_BUCKET']}/{data_object_key}")
    else:
        import numpy as np
        from sklearn.datasets import make_classification

        np.random.seed(42)
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=6, n_redundant=2, random_state=42
        )
        feature_names = [
            "credit_score", "age", "tenure", "balance",
            "num_products", "has_credit_card", "is_active_member",
            "estimated_salary", "geography_encoded", "gender_encoded",
        ]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        target_column = "target"
        print(f"No S3 connection attached — using the synthetic dataset ({len(df):,} rows)")

    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' not found. Available: {list(df.columns)}"
        )

    # ----------------------------------------------------------------- train
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    feature_columns = [c for c in df.columns if c != target_column]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(train_df[feature_columns], train_df[target_column])
    print(f"Test accuracy: {model.score(test_df[feature_columns], test_df[target_column]):.4f}")

    # -------------------------------------------------------------- validate
    vm_train_ds = vm.init_dataset(
        dataset=train_df, input_id="train_dataset", target_column=target_column
    )
    vm_test_ds = vm.init_dataset(
        dataset=test_df, input_id="test_dataset", target_column=target_column
    )
    vm_model = vm.init_model(model=model, input_id="gradient_boosting_model")
    vm_train_ds.assign_predictions(model=vm_model)
    vm_test_ds.assign_predictions(model=vm_model)

    for test_id, inputs in [
        ("validmind.data_validation.DatasetDescription", {"dataset": vm_train_ds}),
        ("validmind.data_validation.ClassImbalance", {"dataset": vm_train_ds}),
        ("validmind.model_validation.sklearn.ConfusionMatrix", {"dataset": vm_test_ds, "model": vm_model}),
        ("validmind.model_validation.sklearn.ROCCurve", {"dataset": vm_test_ds, "model": vm_model}),
        ("validmind.model_validation.sklearn.FeatureImportance", {"dataset": vm_train_ds, "model": vm_model}),
    ]:
        result = vm.tests.run_test(
            test_id, inputs=inputs, show=False, generate_description=False
        )
        result.log()
        print(f"logged {test_id}")

    vm.run_test_suite(
        "classifier_full_suite",
        inputs={
            "dataset": vm_test_ds,
            "datasets": [vm_train_ds, vm_test_ds],
            "model": vm_model,
            "train_dataset": vm_train_ds,
            "test_dataset": vm_test_ds,
        },
    )
    print("Full test suite completed and results sent to the ValidMind Platform.")
    return "ok"


@dsl.pipeline(
    name="validmind-validation",
    description="Runs the ValidMind quickstart validation (train reference model, "
    "run tests, upload results) — schedulable and triggerable on demand.",
)
def validmind_validation_pipeline(
    data_object_key: str = "data/training_data.csv",
    target_column: str = "target",
    document: str = "",
    monitoring: bool = False,
):
    task = run_validmind_validation(
        data_object_key=data_object_key,
        target_column=target_column,
        document=document,
        monitoring=monitoring,
    )
    task.set_caching_options(False)  # a validation run must actually run
    kubernetes.use_secret_as_env(
        task,
        secret_name=VALIDMIND_SECRET_NAME,
        secret_key_to_env={k: k for k in VM_ENV_KEYS},
    )
    if S3_SECRET_NAME:
        kubernetes.use_secret_as_env(
            task,
            secret_name=S3_SECRET_NAME,
            secret_key_to_env={k: k for k in S3_ENV_KEYS},
        )


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__) or ".", "dist", "validmind_pipeline.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    compiler.Compiler().compile(validmind_validation_pipeline, out)
    print(f"compiled → {out}")
