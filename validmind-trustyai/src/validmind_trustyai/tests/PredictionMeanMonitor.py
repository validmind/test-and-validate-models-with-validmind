from typing import Optional

from validmind import tags, tasks, test

from ._utils import fairness_row, get_client, require


@test("trustyai.PredictionMeanMonitor")
@tags("trustyai", "monitoring", "drift", "openshift_ai")
@tasks("classification", "regression")
def PredictionMeanMonitor(
    model_id: Optional[str] = None,
    column_name: Optional[str] = None,
    batch_size: int = 250,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Monitors the mean of a model output (or input) column over recent
    inferences and checks it stays inside an expected operating band.

    ### Purpose

    The single most-watched number for a deployed decision model is its output
    rate — approval rate, fraud-flag rate, average score. A move in that rate
    is often the first observable symptom of drift, an upstream data bug, or a
    behavioral change after redeployment, and it is directly explainable to a
    validator or regulator. This test reads the live rate from TrustyAI and
    documents it against the band the model was validated for; run on a
    schedule, it accumulates the metric into an over-time series in ValidMind's
    ongoing-monitoring view.

    ### Test Mechanism

    Calls TrustyAI's identity/batch-mean endpoint (`/metrics/identity`), which
    averages `column_name` over the last `batch_size` inferences. Passes when
    the mean lies within [`lower_bound`, `upper_bound`]; with no bounds given,
    the value is recorded as evidence without a pass/fail verdict.

    ### Signs of High Risk

    - The mean outside the validated operating band (test fails)
    - A steady trend toward a bound across scheduled runs
    - An abrupt step change after a model or data pipeline deployment
    """
    client = get_client(trustyai_url, trustyai_token)
    response = client.compute_column_mean(
        require(model_id, "model_id", "the inference model's name in TrustyAI (see GET /info)"),
        require(column_name, "column_name", "the output/input column to average, e.g. the decision column"),
        batch_size=batch_size,
        lower_threshold=lower_bound,
        upper_threshold=upper_bound,
    )
    value = float(response["value"])

    row = fairness_row(response, **{"Model": model_id, "Column": column_name, "Batch Size": batch_size})
    tables = {f"Mean of '{column_name}' over recent inferences (TrustyAI)": [row]}

    if lower_bound is None and upper_bound is None:
        return tables, value

    thresholds = response.get("thresholds") or {}
    if "outsideBounds" in thresholds:
        passed = not thresholds["outsideBounds"]
    else:
        passed = (lower_bound is None or value >= lower_bound) and (
            upper_bound is None or value <= upper_bound
        )
    return tables, value, passed
