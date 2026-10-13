from typing import Optional

from validmind import tags, tasks, test

from ._utils import get_client


@test("trustyai.InferenceDataHealth")
@tags("trustyai", "data_quality", "monitoring", "openshift_ai")
@tasks("classification", "regression")
def InferenceDataHealth(
    model_id: Optional[str] = None,
    min_observations: int = 100,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Verifies that TrustyAI is actually receiving inference data from the
    deployed model, and that enough observations exist for its metrics to mean
    anything.

    ### Purpose

    Every TrustyAI-backed fairness or drift number is computed over the
    inferences the service has captured. If payload logging is mis-wired — a
    common failure after a redeploy — the monitoring dashboards go quiet and
    every downstream metric is stale or computed over a handful of rows. This
    test documents that the monitoring pipeline itself is healthy, which is a
    precondition for trusting any other TrustyAI evidence in the report.

    ### Test Mechanism

    Reads `GET /info` from the TrustyAI service. For the given `model_id` (or
    every model the service knows, when omitted) it records the observation
    count and the input/output schema sizes, and passes only when each model
    has at least `min_observations` captured inferences.

    ### Signs of High Risk

    - Zero or low observation counts — payload logging is broken or the model
      is not receiving traffic
    - A model missing entirely from `/info` after deployment
    """
    client = get_client(trustyai_url, trustyai_token)
    info = client.info()
    if model_id is not None:
        info = {model_id: client.model_info(model_id)}
    elif not info:
        raise ValueError(
            "TrustyAI reports no models with inference data. Check that payload "
            "logging is configured for your InferenceService."
        )

    rows = []
    passed = True
    for name, entry in sorted(info.items()):
        data = entry.get("data", entry)
        observations = data.get("observations", 0)
        input_items = (data.get("inputSchema") or {}).get("items") or {}
        output_items = (data.get("outputSchema") or {}).get("items") or {}
        ok = observations >= min_observations
        passed = passed and ok
        rows.append(
            {
                "Model": name,
                "Observations": observations,
                "Input Columns": len(input_items),
                "Output Columns": len(output_items),
                "Meets Minimum (%d)" % min_observations: ok,
            }
        )

    tables = {"TrustyAI inference data health": rows}
    metric = float(rows[0]["Observations"]) if model_id is not None else float(
        sum(r["Observations"] for r in rows)
    )
    return tables, metric, passed
