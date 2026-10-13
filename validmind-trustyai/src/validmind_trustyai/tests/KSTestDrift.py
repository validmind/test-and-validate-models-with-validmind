from typing import List, Optional

from validmind import tags, tasks, test

from ._utils import run_drift_metric


@test("trustyai.KSTestDrift")
@tags("trustyai", "drift", "data_quality", "openshift_ai")
@tasks("classification", "regression")
def KSTestDrift(
    model_id: Optional[str] = None,
    reference_tag: str = "TRAINING",
    fit_columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    batch_size: Optional[int] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Detects per-column distribution drift between a deployed model's recent
    inference inputs and its tagged reference data, via TrustyAI's
    Kolmogorov–Smirnov test.

    ### Purpose

    The KS test compares whole distributions, not just means, so it catches
    drift that MeanShift misses: variance changes, emerging bimodality, or tail
    shifts with a stable average. Together with `trustyai.MeanShiftDrift` it
    gives a defensible answer to "is the model still seeing the data it was
    validated on?" — computed by the platform's own monitoring service.

    ### Test Mechanism

    Calls `/metrics/drift/kstest`, which runs a two-sample KS test per column
    between the data tagged `reference_tag` and the most recent inferences. A
    column drifts when its p-value falls below `alpha` (default 0.05). The
    logged metric is the smallest p-value across columns.

    ### Signs of High Risk

    - Any column flagged as drifted (test fails)
    - KS flagging columns MeanShift does not — shape change with a stable mean
    """
    return run_drift_metric(
        "kstest",
        f"Kolmogorov–Smirnov drift of '{model_id}' vs '{reference_tag}' (TrustyAI)",
        model_id, reference_tag, fit_columns, alpha, batch_size,
        trustyai_url, trustyai_token,
    )
