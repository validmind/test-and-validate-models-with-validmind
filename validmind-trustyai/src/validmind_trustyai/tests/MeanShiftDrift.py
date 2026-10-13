from typing import List, Optional

from validmind import tags, tasks, test

from ._utils import run_drift_metric


@test("trustyai.MeanShiftDrift")
@tags("trustyai", "drift", "data_quality", "openshift_ai")
@tasks("classification", "regression")
def MeanShiftDrift(
    model_id: Optional[str] = None,
    reference_tag: str = "TRAINING",
    fit_columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    batch_size: Optional[int] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Detects per-column mean drift between a deployed model's recent inference
    inputs and its tagged reference (training) data, via TrustyAI's MeanShift
    metric.

    ### Purpose

    A shift in a feature's mean is the simplest, most interpretable form of data
    drift: the population the model now serves no longer looks like the one it
    was trained on, and its performance guarantees no longer apply. This test
    turns TrustyAI's production drift monitoring into documented validation
    evidence in ValidMind.

    ### Test Mechanism

    Calls `/metrics/drift/meanshift`, which runs a t-test per column between the
    data tagged `reference_tag` (upload it once with TrustyAI's `/data/upload`)
    and the most recent inferences. A column drifts when its p-value falls below
    `alpha` (default 0.05). The logged metric is the smallest p-value, so a
    scheduled run builds an over-time drift series.

    ### Signs of High Risk

    - Any column flagged as drifted (test fails)
    - p-values trending downward across runs — drift building before it triggers
    """
    return run_drift_metric(
        "meanshift",
        f"MeanShift drift of '{model_id}' vs '{reference_tag}' (TrustyAI)",
        model_id, reference_tag, fit_columns, alpha, batch_size,
        trustyai_url, trustyai_token,
    )
