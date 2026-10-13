from typing import List, Optional

from validmind import tags, tasks, test

from ._utils import get_client, require


@test("trustyai.ScheduledMetricCoverage")
@tags("trustyai", "monitoring", "governance", "openshift_ai")
@tasks("classification", "regression")
def ScheduledMetricCoverage(
    model_id: Optional[str] = None,
    expected_metrics: Optional[List[str]] = None,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Verifies that recurring TrustyAI metric schedules actually exist for the
    deployed model — that the monitoring promised in the model's documentation
    is really running.

    ### Purpose

    Model governance usually *requires* continuous bias and drift monitoring,
    and reviewers sign off based on that promise. But TrustyAI schedules are
    API state, not declarative config: they disappear when a service is
    recreated and nothing alerts on their absence. This test closes the gap
    between "monitoring is documented" and "monitoring is configured" by
    listing the live schedules and failing when an expected metric has none.

    ### Test Mechanism

    Reads `GET /metrics/all/requests` from the TrustyAI service, filters to
    `model_id`, and records every active schedule (metric name, request id,
    and, for fairness metrics, the audited attribute and outcome). Passes when
    each name in `expected_metrics` (e.g. `["SPD", "MEANSHIFT"]`) has at least
    one active schedule. The logged metric is the number of active schedules.

    ### Signs of High Risk

    - An expected metric with zero schedules — monitoring silently dropped,
      e.g. after the TrustyAI service or its storage was recreated
    - Schedules pointing at stale model ids after a redeploy
    """
    client = get_client(trustyai_url, trustyai_token)
    require(model_id, "model_id", "the inference model's name in TrustyAI (see GET /info)")
    expected = [m.upper() for m in (expected_metrics or ["SPD"])]

    requests_ = [r for r in client.scheduled_requests() if r.get("modelId") == model_id]

    def metric_name(entry):
        # Java entries embed the original request; Python entries are flat.
        return (entry.get("metricName") or entry.get("request", {}).get("metricName") or "").upper()

    schedule_rows = [
        {
            "Metric": metric_name(r) or "<unknown>",
            "Request ID": r.get("requestId") or r.get("id"),
            "Protected Attribute": r.get("protectedAttribute")
            or r.get("request", {}).get("protectedAttribute"),
            "Outcome": r.get("outcomeName") or r.get("request", {}).get("outcomeName"),
            "Batch Size": r.get("batchSize") or r.get("request", {}).get("batchSize"),
        }
        for r in requests_
    ]
    active_names = {row["Metric"] for row in schedule_rows}
    coverage_rows = [
        {"Expected Metric": name, "Scheduled": name in active_names} for name in expected
    ]
    passed = all(row["Scheduled"] for row in coverage_rows)

    tables = {
        f"Recurring metric coverage for '{model_id}'": coverage_rows,
        "Active TrustyAI schedules": schedule_rows
        or [{"Metric": "<none>", "Request ID": None, "Protected Attribute": None, "Outcome": None, "Batch Size": None}],
    }
    return tables, float(len(schedule_rows)), passed
