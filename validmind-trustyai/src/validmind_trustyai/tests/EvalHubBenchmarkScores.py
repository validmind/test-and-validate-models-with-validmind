from typing import Dict, Optional

from validmind import tags, tasks, test

from validmind_trustyai.evalhub import MLflowReader, run_metrics, run_tags


@test("trustyai.EvalHubBenchmarkScores")
@tags("trustyai", "evalhub", "llm", "model_performance", "openshift_ai")
@tasks("text_qa", "text_summarization")
def EvalHubBenchmarkScores(
    experiment_name: Optional[str] = None,
    metric_thresholds: Optional[Dict[str, float]] = None,
    filter_tags: Optional[Dict[str, str]] = None,
    max_runs: int = 100,
    mlflow_uri: Optional[str] = None,
    mlflow_token: Optional[str] = None,
):
    """Reads the latest EvalHub benchmark scores from MLflow, checks them
    against minimum thresholds, and logs them as validation evidence.

    ### Purpose

    EvalHub jobs fan out one benchmark run per metric family, and the provider
    logs the resulting scores to MLflow. Those scores are the quantitative
    evidence behind statements like "the assistant meets our bias and
    grounding bars" — but they live in an experiment-tracking tool, outside
    the governance record. This test pulls the most recent score for each
    thresholded metric into ValidMind, applies the documented bar, and (run on
    a schedule) accumulates score-over-time series in the ongoing-monitoring
    view.

    ### Test Mechanism

    Resolves `experiment_name` in MLflow, searches its runs (newest first,
    optionally narrowed by `filter_tags`, e.g. an assistant or session id),
    and takes the most recent value of every metric named in
    `metric_thresholds`. Passes when each metric meets or exceeds its
    threshold; a thresholded metric with no run at all fails. The logged
    metric is the worst margin over threshold (negative = below the bar).

    ### Signs of High Risk

    - Any score below its threshold, or a thresholded metric that no recent
      run reports
    - Margins shrinking across scheduled runs — quality drifting toward the
      bar before it breaks
    """
    if not experiment_name:
        raise ValueError(
            "Param 'experiment_name' is required — the MLflow experiment EvalHub "
            "logs benchmark runs to (e.g. 'gaussia-agent-evaluation')"
        )
    if not metric_thresholds:
        raise ValueError(
            "Param 'metric_thresholds' is required — {metric_name: minimum_score}, "
            "e.g. {'humanity_score': 0.7, 'toxicity_pass_rate': 0.95}"
        )

    reader = MLflowReader(mlflow_uri, mlflow_token)
    experiment_id = reader.experiment_id(experiment_name)
    filter_string = " and ".join(
        f"tags.`{key}` = '{value}'" for key, value in (filter_tags or {}).items()
    )
    runs = reader.search_runs(experiment_id, filter_string, max_results=max_runs)
    if not runs:
        raise ValueError(
            f"No MLflow runs found in experiment '{experiment_name}'"
            + (f" matching {filter_tags}" if filter_tags else "")
        )

    # Runs come newest-first: the first run reporting a metric wins.
    latest: Dict[str, Dict] = {}
    for run in runs:
        metrics = run_metrics(run)
        for key in metric_thresholds:
            if key in metrics and key not in latest:
                latest[key] = {
                    "value": metrics[key],
                    "run": run.get("info", {}).get("run_name")
                    or run.get("info", {}).get("run_id", ""),
                    "benchmark": run_tags(run).get("benchmark", ""),
                }

    rows = []
    worst_margin = None
    passed = True
    for key, threshold in sorted(metric_thresholds.items()):
        entry = latest.get(key)
        value = entry["value"] if entry else None
        margin = (value - threshold) if value is not None else None
        ok = margin is not None and margin >= 0
        passed = passed and ok
        if margin is not None and (worst_margin is None or margin < worst_margin):
            worst_margin = margin
        rows.append(
            {
                "Metric": key,
                "Latest Score": value,
                "Threshold (min)": threshold,
                "Margin": margin,
                "Meets Bar": ok,
                "Run": entry["run"] if entry else "<no run reports this metric>",
            }
        )

    tables = {f"EvalHub benchmark scores vs thresholds ('{experiment_name}')": rows}
    return tables, (worst_margin if worst_margin is not None else 0.0), passed
