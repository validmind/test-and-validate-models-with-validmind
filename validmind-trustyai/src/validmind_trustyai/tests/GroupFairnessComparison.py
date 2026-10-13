from typing import List, Optional

from validmind import tags, tasks, test

from ._utils import get_client, require


@test("trustyai.GroupFairnessComparison")
@tags("trustyai", "fairness", "bias", "openshift_ai")
@tasks("classification")
def GroupFairnessComparison(
    model_id: Optional[str] = None,
    protected_attribute: Optional[str] = None,
    group_pairs: Optional[List[List]] = None,
    outcome_name: Optional[str] = None,
    favorable_outcome=None,
    spd_threshold: float = 0.1,
    dir_threshold: float = 0.2,
    batch_size: Optional[int] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Audits fairness across multiple group pairings of a protected attribute in
    one pass, computing both SPD and DIR from TrustyAI's live inference data.

    ### Purpose

    Single privileged/unprivileged comparisons hide problems when a protected
    attribute has more than two values (for example, multiple age bands or
    geographies). This test sweeps a list of group pairings and reports
    Statistical Parity Difference and Disparate Impact Ratio side by side for
    each, so a disparity affecting any one group is surfaced — and documented —
    rather than averaged away.

    ### Test Mechanism

    For every `[privileged, unprivileged]` pair in `group_pairs`, requests SPD
    and DIR from the TrustyAI service over the same batch of recent inferences.
    The test passes only when every pairing is inside both fair bands
    (SPD ±`spd_threshold`, DIR 1 ±`dir_threshold`). The reported metric is the
    worst absolute SPD across pairings.

    ### Signs of High Risk

    - Any pairing outside either fair band (test fails)
    - Large spread between pairings: overall parity but localized disadvantage
    """
    client = get_client(trustyai_url, trustyai_token)
    require(model_id, "model_id", "the inference model's name in TrustyAI (see GET /info)")
    require(protected_attribute, "protected_attribute", "input column to audit")
    require(outcome_name, "outcome_name", "output column holding the decision")
    require(favorable_outcome, "favorable_outcome", "output value considered favorable")
    if not group_pairs:
        raise ValueError(
            "Param 'group_pairs' is required — a list of [privileged, unprivileged] "
            "value pairs, e.g. [[1, 0]] or [['30-40', 'under-30'], ['30-40', 'over-60']]"
        )

    rows = []
    worst_spd = 0.0
    all_within = True
    for pair in group_pairs:
        privileged, unprivileged = pair
        spd = client.compute_group_metric(
            "spd", model_id, protected_attribute, privileged, unprivileged,
            outcome_name, favorable_outcome, batch_size=batch_size,
            threshold_delta=spd_threshold,
        )
        dir_ = client.compute_group_metric(
            "dir", model_id, protected_attribute, privileged, unprivileged,
            outcome_name, favorable_outcome, batch_size=batch_size,
            threshold_delta=dir_threshold,
        )
        spd_ok = not (spd.get("thresholds") or {}).get("outsideBounds", False)
        dir_ok = not (dir_.get("thresholds") or {}).get("outsideBounds", False)
        rows.append(
            {
                "Privileged": privileged,
                "Unprivileged": unprivileged,
                "SPD": spd.get("value"),
                "SPD Within ±%.2f" % spd_threshold: spd_ok,
                "DIR": dir_.get("value"),
                "DIR Within 1±%.2f" % dir_threshold: dir_ok,
            }
        )
        worst_spd = max(worst_spd, abs(spd.get("value") or 0.0))
        all_within = all_within and spd_ok and dir_ok

    title = f"Fairness of '{model_id}' across '{protected_attribute}' groups (TrustyAI)"
    return {title: rows}, worst_spd, all_within
