"""Unit tests for the TrustyAI client (HTTP mocked with `responses`).

Run with: pip install -e .[dev] && pytest
"""

import pytest
import responses

from validmind_trustyai.client import TrustyAIClient, TrustyAIError, metric_values

BASE = "https://trustyai.example.com"


def client():
    return TrustyAIClient(base_url=BASE, token="test-token", verify=False)


@responses.activate
def test_compute_spd_sends_expected_payload():
    responses.add(
        responses.POST,
        f"{BASE}/metrics/group/fairness/spd",
        json={
            "type": "metric",
            "name": "SPD",
            "value": -0.05,
            "timestamp": "2026-01-01T00:00:00Z",
            "thresholds": {"lowerBound": -0.1, "upperBound": 0.1, "outsideBounds": False},
        },
    )
    out = client().compute_group_metric(
        "spd", "loan-model", "gender", 1, 0, "approved", 1, batch_size=500
    )
    assert out["value"] == -0.05
    body = responses.calls[0].request
    assert body.headers["Authorization"] == "Bearer test-token"
    import json

    payload = json.loads(body.body)
    assert payload == {
        "modelId": "loan-model",
        "protectedAttribute": "gender",
        "privilegedAttribute": 1,
        "unprivilegedAttribute": 0,
        "outcomeName": "approved",
        "favorableOutcome": 1,
        "batchSize": 500,
    }


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        client().compute_group_metric("nope", "m", "a", 1, 0, "o", 1)
    with pytest.raises(ValueError):
        client().compute_drift("nope", "m", "TRAINING")


@responses.activate
def test_http_error_raises_trustyai_error():
    responses.add(responses.GET, f"{BASE}/info", status=403, body="forbidden")
    with pytest.raises(TrustyAIError, match="403"):
        client().info()


@responses.activate
def test_model_info_lists_known_models_on_miss():
    responses.add(responses.GET, f"{BASE}/info", json={"other-model": {"data": {}}})
    with pytest.raises(TrustyAIError, match="other-model"):
        client().model_info("missing-model")


@responses.activate
def test_scheduled_requests_unwraps_envelope():
    responses.add(
        responses.GET,
        f"{BASE}/metrics/all/requests",
        json={"requests": [{"requestId": "abc", "metricName": "SPD", "modelId": "m"}]},
    )
    out = client().scheduled_requests()
    assert out == [{"requestId": "abc", "metricName": "SPD", "modelId": "m"}]


def test_metric_values_prefers_named_values():
    assert metric_values({"name": "MEANSHIFT", "namedValues": {"age": 0.9}}) == {"age": 0.9}
    assert metric_values({"name": "SPD", "value": 0.1, "namedValues": None}) == {"SPD": 0.1}


def test_missing_url_raises(monkeypatch):
    monkeypatch.delenv("TRUSTYAI_URL", raising=False)
    with pytest.raises(TrustyAIError, match="TRUSTYAI_URL"):
        TrustyAIClient()
