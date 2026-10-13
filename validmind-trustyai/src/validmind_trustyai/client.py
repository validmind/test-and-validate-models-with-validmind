"""Minimal HTTP client for a TrustyAI service on Red Hat OpenShift AI.

Speaks only the endpoints that are stable across the Java (RHOAI 2.x) and
Python (RHOAI 3.x) generations of the service:

- ``GET /info`` — per-model schema, name mappings, and observation counts
- ``POST /metrics/group/fairness/{spd,dir}`` — compute a fairness metric now
- ``POST /metrics/drift/{meanshift,kstest,fouriermmd}`` — compute drift now
- ``POST /metrics/identity`` — mean of a column over recent inferences
- ``GET /metrics/all/requests`` — list scheduled (recurring) metric requests

Configuration comes from arguments or the environment:

- ``TRUSTYAI_URL`` — service base URL. In-cluster this defaults to
  ``https://trustyai-service.<namespace>.svc.cluster.local`` using the pod's
  mounted namespace.
- ``TRUSTYAI_TOKEN`` — bearer token. Defaults to the pod's mounted
  ServiceAccount token when present (any token with ``view`` access to the
  namespace works).
- ``TRUSTYAI_CA_BUNDLE`` — path to a CA bundle, ``TRUSTYAI_VERIFY_SSL=false``
  to disable verification (self-signed demo clusters). In-cluster the mounted
  service CA is used automatically when present.
"""

import os
from typing import Any, Dict, List, Optional, Union

import requests

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
_SA_TOKEN = os.path.join(_SA_DIR, "token")
_SA_NAMESPACE = os.path.join(_SA_DIR, "namespace")
_SA_SERVICE_CA = os.path.join(_SA_DIR, "service-ca.crt")

#: value of a metric response's ``name`` field, keyed by endpoint suffix
GROUP_FAIRNESS_METRICS = {"spd": "SPD", "dir": "DIR"}
DRIFT_METRICS = ("meanshift", "kstest", "fouriermmd")


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


class TrustyAIError(RuntimeError):
    """Raised when the TrustyAI service returns an error response."""


class TrustyAIClient:
    """Small requests-based client for the TrustyAI service REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        verify: Union[bool, str, None] = None,
        timeout: int = 60,
    ):
        base_url = base_url or os.environ.get("TRUSTYAI_URL")
        if not base_url:
            namespace = _read(_SA_NAMESPACE)
            if namespace:
                base_url = f"https://trustyai-service.{namespace}.svc.cluster.local"
        if not base_url:
            raise TrustyAIError(
                "No TrustyAI service URL. Set TRUSTYAI_URL (the trustyai-service "
                "route or in-cluster service URL) or pass base_url=."
            )
        self.base_url = base_url.rstrip("/")

        self.token = token or os.environ.get("TRUSTYAI_TOKEN") or _read(_SA_TOKEN)

        if verify is None:
            if os.environ.get("TRUSTYAI_VERIFY_SSL", "").lower() in ("0", "false", "no"):
                verify = False
            elif os.environ.get("TRUSTYAI_CA_BUNDLE"):
                verify = os.environ["TRUSTYAI_CA_BUNDLE"]
            elif "svc.cluster.local" in self.base_url and os.path.exists(_SA_SERVICE_CA):
                verify = _SA_SERVICE_CA
            else:
                verify = True
        self.verify = verify
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, json: Optional[dict] = None) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.request(
            method,
            f"{self.base_url}{path}",
            json=json,
            headers=headers,
            verify=self.verify,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise TrustyAIError(
                f"TrustyAI {method} {path} failed ({resp.status_code}): {resp.text[:500]}"
            )
        content_type = resp.headers.get("Content-Type", "")
        return resp.json() if "json" in content_type else resp.text

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: dict) -> Any:
        return self._request("POST", path, json=payload)

    # ------------------------------------------------------------------ info
    def info(self) -> Dict[str, Any]:
        """Per-model metadata: input/output schema, name mappings, observations."""
        raw = self.get("/info")
        # The Java service returns a dict keyed by model id; some versions wrap
        # entries in a list — normalize to {model_id: entry}.
        if isinstance(raw, list):
            return {e.get("modelId", str(i)): e for i, e in enumerate(raw)}
        return raw

    def model_info(self, model_id: str) -> Dict[str, Any]:
        info = self.info()
        if model_id not in info:
            known = ", ".join(sorted(info)) or "<none — is the model wired to TrustyAI?>"
            raise TrustyAIError(
                f"Model '{model_id}' unknown to TrustyAI. Models with observed "
                f"inference data: {known}"
            )
        return info[model_id]

    # --------------------------------------------------------------- metrics
    def compute_group_metric(
        self,
        metric: str,
        model_id: str,
        protected_attribute: str,
        privileged: Any,
        unprivileged: Any,
        outcome_name: str,
        favorable_outcome: Any,
        batch_size: Optional[int] = None,
        threshold_delta: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute SPD or DIR now. Returns the standard metric-value response."""
        if metric not in GROUP_FAIRNESS_METRICS:
            raise ValueError(f"metric must be one of {sorted(GROUP_FAIRNESS_METRICS)}")
        payload: Dict[str, Any] = {
            "modelId": model_id,
            "protectedAttribute": protected_attribute,
            "privilegedAttribute": privileged,
            "unprivilegedAttribute": unprivileged,
            "outcomeName": outcome_name,
            "favorableOutcome": favorable_outcome,
        }
        if batch_size is not None:
            payload["batchSize"] = batch_size
        if threshold_delta is not None:
            payload["thresholdDelta"] = threshold_delta
        return self.post(f"/metrics/group/fairness/{metric}", payload)

    def compute_drift(
        self,
        metric: str,
        model_id: str,
        reference_tag: str,
        fit_columns: Optional[List[str]] = None,
        threshold_delta: Optional[float] = None,
        batch_size: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute a drift metric now against data tagged ``reference_tag``."""
        if metric not in DRIFT_METRICS:
            raise ValueError(f"metric must be one of {DRIFT_METRICS}")
        payload: Dict[str, Any] = {"modelId": model_id, "referenceTag": reference_tag}
        if fit_columns:
            payload["fitColumns"] = fit_columns
        if threshold_delta is not None:
            payload["thresholdDelta"] = threshold_delta
        if batch_size is not None:
            payload["batchSize"] = batch_size
        if extra:
            payload.update(extra)
        return self.post(f"/metrics/drift/{metric}", payload)

    def compute_column_mean(
        self,
        model_id: str,
        column_name: str,
        batch_size: Optional[int] = None,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Mean of one input/output column over the last ``batch_size`` inferences.

        Uses ``/metrics/identity`` (kept as an alias for ``/metrics/batchmean``
        in RHOAI 3.x).
        """
        payload: Dict[str, Any] = {"modelId": model_id, "columnName": column_name}
        if batch_size is not None:
            payload["batchSize"] = batch_size
        if lower_threshold is not None:
            payload["lowerThreshold"] = lower_threshold
        if upper_threshold is not None:
            payload["upperThreshold"] = upper_threshold
        return self.post("/metrics/identity", payload)

    def scheduled_requests(self, metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """All scheduled (recurring) metric requests known to the service."""
        path = "/metrics/all/requests"
        if metric_type:
            path += f"?type={metric_type}"
        raw = self.get(path)
        return raw.get("requests", raw) if isinstance(raw, dict) else raw


def metric_values(response: Dict[str, Any]) -> Dict[str, float]:
    """Normalize a metric response to {series_name: value}.

    Fairness/identity responses carry a single ``value``; drift responses
    carry per-column ``namedValues``.
    """
    named = response.get("namedValues")
    if named:
        return dict(named)
    return {response.get("name", "value"): response.get("value")}
