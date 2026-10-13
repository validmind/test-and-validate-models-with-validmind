from typing import Optional

from validmind import tags, tasks, test

from ._utils import fairness_row, get_client, require


@test("trustyai.StatisticalParityDifference")
@tags("trustyai", "fairness", "bias", "openshift_ai")
@tasks("classification")
def StatisticalParityDifference(
    model_id: Optional[str] = None,
    protected_attribute: Optional[str] = None,
    privileged_group=None,
    unprivileged_group=None,
    outcome_name: Optional[str] = None,
    favorable_outcome=None,
    threshold_delta: float = 0.1,
    batch_size: Optional[int] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Checks group fairness of a deployed model by reading the Statistical Parity
    Difference from TrustyAI's live inference data.

    ### Purpose

    Statistical Parity Difference (SPD) is the difference in the rate of favorable
    outcomes between the unprivileged and privileged groups of a protected
    attribute. A model that treats both groups alike scores near 0; the widely
    used fair band is ±`threshold_delta` (default ±0.1).

    Unlike an offline fairness test computed on a static dataset, this test asks
    the TrustyAI service deployed alongside the model on OpenShift AI, so the
    value reflects the inferences the model has actually served in production.

    ### Test Mechanism

    Sends a compute request to `/metrics/group/fairness/spd` for the last
    `batch_size` observed inferences and records the returned value, the fair
    bounds, and TrustyAI's plain-language interpretation.

    ### Signs of High Risk

    - SPD outside the ±`threshold_delta` band (test fails)
    - SPD drifting toward a bound across scheduled runs, even while still inside it
    """
    client = get_client(trustyai_url, trustyai_token)
    response = client.compute_group_metric(
        "spd",
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
    tables = {"Statistical Parity Difference (TrustyAI)": [row]}
    interpretation = response.get("specificDefinition")
    if interpretation:
        tables["Interpretation"] = [{"TrustyAI Interpretation": interpretation}]

    return tables, float(response["value"]), passed
