from typing import List, Optional

from validmind import tags, tasks, test

from ._utils import run_drift_metric


@test("trustyai.FourierMMDDrift")
@tags("trustyai", "drift", "data_quality", "openshift_ai")
@tasks("classification", "regression")
def FourierMMDDrift(
    model_id: Optional[str] = None,
    reference_tag: str = "TRAINING",
    fit_columns: Optional[List[str]] = None,
    alpha: float = 0.05,
    batch_size: Optional[int] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Detects multivariate drift in a deployed model's inference inputs using
    TrustyAI's FourierMMD (Maximum Mean Discrepancy) metric.

    ### Purpose

    Column-by-column tests miss drift in the *relationships* between features —
    for instance, income and age each staying in range while their correlation
    inverts. FourierMMD compares the joint distribution of recent inferences
    against the tagged reference data, closing the gap left by the univariate
    MeanShift and KS tests. Running all three gives layered drift evidence for
    the validation report.

    ### Test Mechanism

    Calls `/metrics/drift/fouriermmd`, which estimates the maximum mean
    discrepancy between the reference window and recent inferences using random
    Fourier features. Drift is flagged when the returned significance falls
    below `alpha`.

    ### Signs of High Risk

    - Drift flagged here while the univariate tests pass — a joint-distribution
      change that per-column monitoring cannot see
    """
    return run_drift_metric(
        "fouriermmd",
        f"FourierMMD multivariate drift of '{model_id}' vs '{reference_tag}' (TrustyAI)",
        model_id, reference_tag, fit_columns, alpha, batch_size,
        trustyai_url, trustyai_token,
    )
