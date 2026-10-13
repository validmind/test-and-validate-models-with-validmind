"""Unit tests for the EvalHub-backed tests (MLflow HTTP mocked)."""

import pytest
import responses

import validmind_trustyai  # noqa: F401 — registers the trustyai.* tests
from validmind.tests import run_test
from validmind_trustyai.evalhub import EvalHubError, evalhub_client

MLFLOW = "https://mlflow.example.com"


def mlflow_run(name, metrics, tags=None):
    return {
        "info": {"run_id": name, "run_name": name},
        "data": {
            "metrics": [{"key": k, "value": v} for k, v in metrics.items()],
            "tags": [{"key": k, "value": v} for k, v in (tags or {}).items()],
        },
    }


@responses.activate
def test_benchmark_scores_pass_and_fail():
    responses.add(
        responses.GET,
        f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name",
        json={"experiment": {"experiment_id": "7"}},
    )
    responses.add(
        responses.POST,
        f"{MLFLOW}/api/2.0/mlflow/runs/search",
        json={
            "runs": [
                mlflow_run("newest", {"humanity_score": 0.82}),
                mlflow_run("older", {"humanity_score": 0.50, "toxicity_pass_rate": 0.91}),
            ]
        },
    )
    result = run_test(
        "trustyai.EvalHubBenchmarkScores",
        params={
            "experiment_name": "gaussia-agent-evaluation",
            "metric_thresholds": {"humanity_score": 0.7, "toxicity_pass_rate": 0.95},
            "mlflow_uri": MLFLOW,
        },
        generate_description=False,
        show=False,
    )
    # newest humanity_score (0.82) wins over the older run; toxicity 0.91 < 0.95
    assert result.passed is False
    assert result.metric == pytest.approx(0.91 - 0.95)


@responses.activate
def test_benchmark_scores_missing_metric_fails():
    responses.add(
        responses.GET,
        f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name",
        json={"experiment": {"experiment_id": "7"}},
    )
    responses.add(
        responses.POST,
        f"{MLFLOW}/api/2.0/mlflow/runs/search",
        json={"runs": [mlflow_run("r1", {"other_metric": 1.0})]},
    )
    result = run_test(
        "trustyai.EvalHubBenchmarkScores",
        params={
            "experiment_name": "exp",
            "metric_thresholds": {"humanity_score": 0.7},
            "mlflow_uri": MLFLOW,
        },
        generate_description=False,
        show=False,
    )
    assert result.passed is False


def test_evalhub_client_errors_without_sdk_or_url(monkeypatch):
    monkeypatch.delenv("EVALHUB_BASE_URL", raising=False)
    try:
        import evalhub  # noqa: F401

        with pytest.raises(EvalHubError, match="EVALHUB_BASE_URL"):
            evalhub_client()
    except ImportError:
        with pytest.raises(EvalHubError, match="validmind-trustyai\\[evalhub\\]"):
            evalhub_client()


def test_all_twelve_tests_registered():
    assert len(validmind_trustyai.TEST_IDS) == 12
