"""Unit tests for the ValidMind test functions (TrustyAI HTTP mocked).

These call the undecorated logic through validmind's run_test so the
registration path is exercised too.
"""

import responses

import validmind_trustyai  # noqa: F401 — registers the trustyai.* tests
from validmind.tests import run_test

BASE = "https://trustyai.example.com"
AUTH = {"trustyai_url": BASE, "trustyai_token": "t"}


def spd_response(value=-0.02, outside=False):
    return {
        "type": "metric",
        "name": "SPD",
        "value": value,
        "timestamp": "2026-01-01T00:00:00Z",
        "specificDefinition": "The SPD of %f indicates ..." % value,
        "thresholds": {"lowerBound": -0.1, "upperBound": 0.1, "outsideBounds": outside},
    }


@responses.activate
def test_spd_passes_inside_bounds():
    responses.add(responses.POST, f"{BASE}/metrics/group/fairness/spd", json=spd_response())
    result = run_test(
        "trustyai.StatisticalParityDifference",
        params={
            "model_id": "loan-model",
            "protected_attribute": "gender",
            "privileged_group": 1,
            "unprivileged_group": 0,
            "outcome_name": "approved",
            "favorable_outcome": 1,
            **AUTH,
        },
        generate_description=False,
        show=False,
    )
    assert result.passed is True
    assert result.metric == -0.02


@responses.activate
def test_spd_fails_outside_bounds():
    responses.add(
        responses.POST,
        f"{BASE}/metrics/group/fairness/spd",
        json=spd_response(value=-0.3, outside=True),
    )
    result = run_test(
        "trustyai.StatisticalParityDifference",
        params={
            "model_id": "loan-model",
            "protected_attribute": "gender",
            "privileged_group": 1,
            "unprivileged_group": 0,
            "outcome_name": "approved",
            "favorable_outcome": 1,
            **AUTH,
        },
        generate_description=False,
        show=False,
    )
    assert result.passed is False


@responses.activate
def test_kstest_drift_flags_low_p_value():
    responses.add(
        responses.POST,
        f"{BASE}/metrics/drift/kstest",
        json={
            "type": "metric",
            "name": "KSTEST",
            "value": None,
            "namedValues": {"age": 0.001, "income": 0.8},
            "timestamp": "2026-01-01T00:00:00Z",
        },
    )
    result = run_test(
        "trustyai.KSTestDrift",
        params={"model_id": "loan-model", **AUTH},
        generate_description=False,
        show=False,
    )
    assert result.passed is False
    assert result.metric == 0.001


@responses.activate
def test_inference_data_health_minimum():
    responses.add(
        responses.GET,
        f"{BASE}/info",
        json={
            "loan-model": {
                "data": {
                    "observations": 42,
                    "inputSchema": {"items": {"a": {}, "b": {}}},
                    "outputSchema": {"items": {"y": {}}},
                }
            }
        },
    )
    result = run_test(
        "trustyai.InferenceDataHealth",
        params={"model_id": "loan-model", "min_observations": 100, **AUTH},
        generate_description=False,
        show=False,
    )
    assert result.passed is False
    assert result.metric == 42.0


@responses.activate
def test_scheduled_metric_coverage():
    responses.add(
        responses.GET,
        f"{BASE}/metrics/all/requests",
        json={
            "requests": [
                {"requestId": "1", "metricName": "SPD", "modelId": "loan-model"},
            ]
        },
    )
    result = run_test(
        "trustyai.ScheduledMetricCoverage",
        params={"model_id": "loan-model", "expected_metrics": ["SPD", "MEANSHIFT"], **AUTH},
        generate_description=False,
        show=False,
    )
    assert result.passed is False  # MEANSHIFT has no schedule


def test_all_tests_registered():
    assert len(validmind_trustyai.TEST_IDS) == 12
    from validmind.tests import describe_test

    for test_id in validmind_trustyai.TEST_IDS:
        assert describe_test(test_id, show=False) is not None
