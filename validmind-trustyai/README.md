# validmind-trustyai

ValidMind tests that read live bias, drift, and data-health evidence from a
[TrustyAI](https://trustyai.org) service on Red Hat OpenShift AI and log it to
the [ValidMind Platform](https://validmind.com).

TrustyAI *collects* production metrics; ValidMind is the *system of record*
for model documentation, validation, and ongoing monitoring. These tests are
the bridge: each one asks the TrustyAI service deployed next to your model for
a metric, applies a pass/fail policy, and logs the evidence (tables, values,
thresholds) into your model's documentation — so what your validators sign off
on is what production actually shows.

## The tests

Twelve tests, each answering one question a validator actually asks:

| Test id | Question it answers |
|---|---|
| `trustyai.StatisticalParityDifference` | Is the live favorable-outcome rate fair across a protected attribute (SPD within ±0.1)? |
| `trustyai.DisparateImpactRatio` | Does the live outcome ratio satisfy the four-fifths rule (DIR within 0.8–1.2)? |
| `trustyai.GroupFairnessComparison` | Is *every* pairing of a multi-valued protected attribute fair — SPD and DIR per group pair, worst case reported? |
| `trustyai.MeanShiftDrift` | Have per-feature means shifted from the training data (t-test per column)? |
| `trustyai.KSTestDrift` | Have per-feature *distributions* shifted (Kolmogorov–Smirnov per column)? |
| `trustyai.FourierMMDDrift` | Has the *joint* distribution shifted, even if every column looks fine (multivariate MMD)? |
| `trustyai.PredictionMeanMonitor` | Is the model's output rate (approval rate, average score) inside its validated operating band? |
| `trustyai.InferenceDataHealth` | Is TrustyAI actually receiving inference payloads, and enough of them to trust its metrics? |
| `trustyai.SchemaIntegrity` | Does the schema the model serves match the features that were documented and validated? |
| `trustyai.ScheduledMetricCoverage` | Are the recurring TrustyAI metric schedules the documentation promises actually configured? |
| `trustyai.EvalHubRunCoverage` | Are the promised EvalHub evaluations (bias, toxicity, grounding, …) actually being submitted and succeeding? |
| `trustyai.EvalHubBenchmarkScores` | Do the latest EvalHub benchmark scores (read from MLflow) meet their documented thresholds? |

The two `EvalHub*` tests cover the evaluation side of TrustyAI — EvalHub
orchestrates benchmark jobs and its providers log the scores to MLflow — and
need the optional SDK extra: `pip install validmind-trustyai[evalhub]`
(`EvalHubBenchmarkScores` talks straight to the MLflow REST API and works
without the extra).

Every test that returns a scalar (all but `SchemaIntegrity`) is automatically
logged as a ValidMind *unit metric*, so scheduled runs accumulate over-time
series in the ongoing-monitoring view — SPD over time, minimum drift p-value
over time, approval rate over time.

## Install

```bash
pip install validmind-trustyai        # once published
# or, from this repository:
pip install ./validmind-trustyai
```

## Use in a workbench

```python
import validmind as vm
import validmind_trustyai  # registers the trustyai.* tests

vm.init()  # reads VM_API_HOST / VM_API_KEY / VM_API_SECRET / VM_API_MODEL

result = vm.tests.run_test(
    "trustyai.StatisticalParityDifference",
    params={
        "model_id": "loan-model",            # the model's name in TrustyAI (GET /info)
        "protected_attribute": "gender",
        "privileged_group": 1,
        "unprivileged_group": 0,
        "outcome_name": "approved",
        "favorable_outcome": 1,
        # optional when running outside the cluster:
        # "trustyai_url": "https://trustyai-service-<project>.apps.<cluster>",
        # "trustyai_token": "<oc whoami -t>",
    },
)
result.log()
```

## Use headlessly (scheduled pipeline)

```python
import validmind as vm
import validmind_trustyai

vm.init(monitoring=True, document="monitoring", generate_descriptions=False)

for test_id, params in my_test_plan:
    vm.tests.run_test(test_id, params=params, show=False, generate_description=False).log()
```

## TrustyAI connection settings

| Setting | Source (in order) |
|---|---|
| Service URL | `trustyai_url` param → `TRUSTYAI_URL` env → `https://trustyai-service.<pod namespace>.svc.cluster.local` |
| Bearer token | `trustyai_token` param → `TRUSTYAI_TOKEN` env → the pod's mounted ServiceAccount token |
| TLS | `TRUSTYAI_CA_BUNDLE` (path) or `TRUSTYAI_VERIFY_SSL=false`; in-cluster the mounted service CA is used automatically |

Any token with `view` access to the data science project works (`oc whoami -t`,
or a ServiceAccount token in a pipeline).

## EvalHub / MLflow connection settings

| Setting | Source |
|---|---|
| EvalHub URL | `evalhub_url` param → `EVALHUB_BASE_URL` env |
| EvalHub auth | `evalhub_token` param → `EVALHUB_AUTH_TOKEN` / `EVALHUB_AUTH_TOKEN_PATH`; `EVALHUB_TENANT` (default `default`); `EVALHUB_INSECURE=true` for self-signed |
| MLflow URL | `mlflow_uri` param → `MLFLOW_TRACKING_URI` env |
| MLflow auth | `mlflow_token` param → `MLFLOW_TRACKING_TOKEN`; `MLFLOW_VERIFY_SSL=false` for self-signed |

Prerequisites on the OpenShift AI side: TrustyAI enabled for the project, the
model's InferenceService wired for payload logging, and — for the drift tests —
reference data uploaded and tagged (e.g. `TRAINING`) via TrustyAI's
`/data/upload` endpoint. See the [TrustyAI bias](https://trustyai.org/docs/main/bias-monitoring)
and [drift](https://trustyai.org/docs/main/data-drift-monitoring) tutorials.

## Compatibility

Only endpoints stable across the Java (RHOAI 2.x) and Python (RHOAI 3.x)
generations of the TrustyAI service are used: `/info`,
`/metrics/group/fairness/{spd,dir}`, `/metrics/drift/{meanshift,kstest,fouriermmd}`,
`/metrics/identity`, and `/metrics/all/requests`.

## Development

```bash
pip install -e .[dev]
pytest
```
