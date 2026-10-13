"""Shared helpers for the TrustyAI-backed ValidMind tests."""

from typing import Any, Dict, List, Optional

from validmind_trustyai.client import TrustyAIClient


def get_client(trustyai_url: Optional[str] = None, trustyai_token: Optional[str] = None) -> TrustyAIClient:
    """Build a client from explicit params, falling back to the environment."""
    return TrustyAIClient(base_url=trustyai_url, token=trustyai_token)


def require(value: Any, name: str, hint: str) -> Any:
    if value in (None, ""):
        raise ValueError(f"Param '{name}' is required — {hint}")
    return value


def fairness_row(response: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """One result table row from a fairness/identity metric response."""
    thresholds = response.get("thresholds") or {}
    row = {
        "Metric": response.get("name"),
        "Value": response.get("value"),
        "Lower Bound": thresholds.get("lowerBound"),
        "Upper Bound": thresholds.get("upperBound"),
        "Within Bounds": not thresholds.get("outsideBounds", False),
        "Computed At": response.get("timestamp"),
    }
    row.update(extra)
    return row


def run_drift_metric(
    metric: str,
    title: str,
    model_id: Optional[str],
    reference_tag: str,
    fit_columns: Optional[List[str]],
    alpha: float,
    batch_size: Optional[int],
    trustyai_url: Optional[str],
    trustyai_token: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
):
    """Shared body of the drift tests: compute, tabulate, pass/fail.

    Returns (tables, min_p_value, passed) — the smallest per-column p-value is
    the logged metric so scheduled runs build an over-time drift series.
    """
    client = get_client(trustyai_url, trustyai_token)
    response = client.compute_drift(
        metric,
        require(model_id, "model_id", "the inference model's name in TrustyAI (see GET /info)"),
        reference_tag,
        fit_columns=fit_columns,
        threshold_delta=alpha,
        batch_size=batch_size,
        extra=extra,
    )
    rows = drift_rows(response, alpha)
    if not rows:
        raise ValueError(
            f"TrustyAI returned no per-column values for {metric}. Is reference "
            f"data uploaded and tagged '{reference_tag}' for model '{model_id}'?"
        )
    p_values = [r["p-value"] for r in rows if r["p-value"] is not None]
    passed = not any(r["Drift Detected"] for r in rows)
    return {title: rows}, (min(p_values) if p_values else 0.0), passed


def drift_rows(response: Dict[str, Any], alpha: float) -> List[Dict[str, Any]]:
    """Per-column result table rows from a drift metric response.

    Drift responses carry per-column p-values in ``namedValues`` (a single
    ``value`` when the service collapses to one series). p < alpha means the
    column's recent inference distribution differs significantly from the
    tagged reference data.
    """
    named = response.get("namedValues") or {}
    if not named and response.get("value") is not None:
        named = {response.get("name", "all"): response["value"]}
    return [
        {
            "Column": column,
            "p-value": p_value,
            "Drift Detected": (p_value is not None and p_value < alpha),
        }
        for column, p_value in sorted(named.items())
    ]
