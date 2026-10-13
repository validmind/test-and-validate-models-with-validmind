from typing import Optional

from validmind import tags, tasks, test

from ._utils import fairness_row, get_client, require


@test("trustyai.DisparateImpactRatio")
@tags("trustyai", "fairness", "bias", "openshift_ai")
@tasks("classification")
def DisparateImpactRatio(
    model_id: Optional[str] = None,
    protected_attribute: Optional[str] = None,
    privileged_group=None,
    unprivileged_group=None,
    outcome_name: Optional[str] = None,
    favorable_outcome=None,
    threshold_delta: float = 0.2,
    batch_size: Optional[int] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Checks group fairness of a deployed model by reading the Disparate Impact
    Ratio from TrustyAI's live inference data.

    ### Purpose

    Disparate Impact Ratio (DIR) is the rate of favorable outcomes for the
    unprivileged group divided by the rate for the privileged group. Parity is
    1.0; the regulatory rule of thumb (the "four-fifths rule") flags ratios
    below 0.8. TrustyAI's fair band is 1 ± `threshold_delta` (default 0.8–1.2).

    Because the value is computed by the TrustyAI service from inferences the
    deployed model actually served, this is production evidence rather than an
    offline estimate, and complements SPD by expressing the disparity as a
    ratio, which is easier to compare across models with different base rates.

    ### Test Mechanism

    Sends a compute request to `/metrics/group/fairness/dir` for the last
    `batch_size` observed inferences and records the returned value, the fair
    bounds, and TrustyAI's plain-language interpretation.

    ### Signs of High Risk

    - DIR below the lower bound (unprivileged group disadvantaged — the classic
      four-fifths violation) or above the upper bound
    - A ratio computed over very few favorable outcomes; check data volume with
      the `trustyai.InferenceDataHealth` test
    """
    client = get_client(trustyai_url, trustyai_token)
    response = client.compute_group_metric(
        "dir",
        require(model_id, "model_id", "the inference model's name in TrustyAI (see GET /info)"),
        require(protected_attribute, "protected_attribute", "input column to audit, e.g. 'gender'"),
        require(privileged_group, "privileged_group", "attribute value of the privileged group"),
        require(unprivileged_group, "unprivileged_group", "attribute value of the unprivileged group"),
        require(outcome_name, "outcome_name", "output column holding the decision"),
        require(favorable_outcome, "favorable_outcome", "output value considered favorable"),
        batch_size=batch_size,
        threshold_delta=threshold_delta,
    )

    row = fairness_row(
        response,
        **{
            "Model": model_id,
            "Protected Attribute": protected_attribute,
            "Privileged": privileged_group,
            "Unprivileged": unprivileged_group,
        },
    )
    passed = bool(row["Within Bounds"])
    tables = {"Disparate Impact Ratio (TrustyAI)": [row]}
    interpretation = response.get("specificDefinition")
    if interpretation:
        tables["Interpretation"] = [{"TrustyAI Interpretation": interpretation}]

    return tables, float(response["value"]), passed
