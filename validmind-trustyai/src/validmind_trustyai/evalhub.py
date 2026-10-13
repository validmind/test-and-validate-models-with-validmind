"""Clients for the EvalHub evaluation service and its MLflow results store.

EvalHub (TrustyAI's evaluation hub) orchestrates evaluation *jobs*: each job
fans out one benchmark run per metric family to a registered provider, and the
provider logs the resulting metrics to MLflow. Reading evaluation evidence
therefore has two surfaces:

- **EvalHub API** (job orchestration state) — accessed through the official
  ``eval-hub-sdk[client]`` package, installed via the ``evalhub`` extra:
  ``pip install validmind-trustyai[evalhub]``.
- **MLflow REST API** (the benchmark scores) — plain HTTP, no SDK needed.

Configuration comes from arguments or the environment, mirroring the
rh-ai-quickstart EvalHub conventions:

- ``EVALHUB_BASE_URL``, ``EVALHUB_AUTH_TOKEN`` / ``EVALHUB_AUTH_TOKEN_PATH``,
  ``EVALHUB_TENANT`` (default ``default``), ``EVALHUB_INSECURE``
- ``MLFLOW_TRACKING_URI``, ``MLFLOW_TRACKING_TOKEN`` (optional bearer token)
"""

import os
from typing import Any, Dict, List, Optional

import requests


class EvalHubError(RuntimeError):
    """Raised when EvalHub or MLflow cannot be reached or returns an error."""


def evalhub_client(
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    tenant: Optional[str] = None,
):
    """Build a SyncEvalHubClient from args/env (requires the ``evalhub`` extra)."""
    try:
        from evalhub import SyncEvalHubClient
    except ImportError as exc:
        raise EvalHubError(
            "The EvalHub tests need the EvalHub SDK. Install it with: "
            "pip install 'validmind-trustyai[evalhub]' "
            "(or: pip install 'eval-hub-sdk[client]')"
        ) from exc

    base_url = base_url or os.environ.get("EVALHUB_BASE_URL")
    if not base_url:
        raise EvalHubError(
            "No EvalHub URL. Set EVALHUB_BASE_URL (e.g. the evalhub Service or "
            "Route in your project) or pass evalhub_url=."
        )
    return SyncEvalHubClient(
        base_url=base_url,
        auth_token=token or os.environ.get("EVALHUB_AUTH_TOKEN"),
        auth_token_path=os.environ.get("EVALHUB_AUTH_TOKEN_PATH"),
        insecure=os.environ.get("EVALHUB_INSECURE", "false").lower() == "true",
        tenant=tenant or os.environ.get("EVALHUB_TENANT", "default"),
    )


def job_state(job: Any) -> str:
    """Normalize a job's status state to an upper-case string."""
    status = getattr(job, "status", None) or (job.get("status") if isinstance(job, dict) else None)
    state = getattr(status, "state", None) if not isinstance(status, dict) else status.get("state")
    if state is None:
        return "UNKNOWN"
    return str(getattr(state, "value", state)).upper()


def job_details(client, job_id: str) -> Dict[str, Any]:
    """Full job document as a plain dict."""
    job = client.jobs.get(job_id)
    if hasattr(job, "model_dump"):
        return job.model_dump(mode="json", exclude_none=True)
    return dict(job)


class MLflowReader:
    """Tiny read-only MLflow REST client (experiments + run search)."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        token: Optional[str] = None,
        verify: Optional[bool] = None,
        timeout: int = 60,
    ):
        tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise EvalHubError(
                "No MLflow tracking URI. Set MLFLOW_TRACKING_URI (the MLflow "
                "server EvalHub logs results to) or pass mlflow_uri=."
            )
        self.base = tracking_uri.rstrip("/")
        self.token = token or os.environ.get("MLFLOW_TRACKING_TOKEN")
        if verify is None:
            verify = os.environ.get("MLFLOW_VERIFY_SSL", "").lower() not in ("0", "false", "no")
        self.verify = verify
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.request(
            method,
            f"{self.base}/api/2.0/mlflow{path}",
            headers=headers,
            verify=self.verify,
            timeout=self.timeout,
            **kwargs,
        )
        if resp.status_code >= 400:
            raise EvalHubError(
                f"MLflow {method} {path} failed ({resp.status_code}): {resp.text[:400]}"
            )
        return resp.json()

    def experiment_id(self, name: str) -> str:
        data = self._request("GET", "/experiments/get-by-name", params={"experiment_name": name})
        return data["experiment"]["experiment_id"]

    def search_runs(
        self,
        experiment_id: str,
        filter_string: str = "",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Most-recent-first runs; each has info/data(metrics, tags)."""
        data = self._request(
            "POST",
            "/runs/search",
            json={
                "experiment_ids": [experiment_id],
                "filter": filter_string,
                "order_by": ["attributes.start_time DESC"],
                "max_results": max_results,
            },
        )
        return data.get("runs", [])


def run_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    """{metric_key: value} for one MLflow search-runs result."""
    return {
        m["key"]: m["value"]
        for m in (run.get("data", {}).get("metrics") or [])
        if isinstance(m.get("value"), (int, float))
    }


def run_tags(run: Dict[str, Any]) -> Dict[str, str]:
    return {t["key"]: t.get("value", "") for t in (run.get("data", {}).get("tags") or [])}
