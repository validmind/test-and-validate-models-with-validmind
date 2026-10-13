from typing import List, Optional

from validmind import tags, tasks, test

from validmind_trustyai.evalhub import evalhub_client, job_details, job_state


@test("trustyai.EvalHubRunCoverage")
@tags("trustyai", "evalhub", "llm", "monitoring", "governance", "openshift_ai")
@tasks("text_qa", "text_summarization")
def EvalHubRunCoverage(
    model_name: Optional[str] = None,
    expected_benchmarks: Optional[List[str]] = None,
    lookback_jobs: int = 25,
    evalhub_url: Optional[str] = None,
    evalhub_token: Optional[str] = None,
    tenant: Optional[str] = None,
):
    """Verifies that the EvalHub evaluations the model's documentation promises
    are actually being submitted and completing.

    ### Purpose

    An LLM or agent governed through ValidMind typically commits to a set of
    recurring evaluations — benchmark families like bias, toxicity, or
    ground-truth accuracy — run through EvalHub, TrustyAI's evaluation
    orchestrator. But an evaluation regime only exists while something submits
    the jobs and they finish: a broken CI hook or failed provider dies
    silently. This test is the counterpart of `trustyai.ScheduledMetricCoverage`
    for the evaluation side: it documents that the promised evaluations ran,
    recently and successfully.

    ### Test Mechanism

    Lists the most recent `lookback_jobs` EvalHub jobs, keeps those evaluating
    `model_name` (all jobs when omitted), and inspects each job's benchmark
    set and status. Passes only when every benchmark id in
    `expected_benchmarks` appears in at least one **succeeded** job, and no
    inspected job for the model has failed. The logged metric is the number of
    succeeded jobs found for the model.

    ### Signs of High Risk

    - An expected benchmark with no succeeded run in the lookback window —
      the evaluation regime has silently stopped
    - Failed jobs for the model — provider or infrastructure problems that
      leave the model effectively unevaluated
    """
    client = evalhub_client(evalhub_url, evalhub_token, tenant)
    expected = [b.lower() for b in (expected_benchmarks or [])]

    with client:
        jobs = client.jobs.list(limit=lookback_jobs)
        rows = []
        succeeded_benchmarks = set()
        succeeded_jobs = 0
        any_failed = False
        for job in jobs:
            details = job_details(client, job.id)
            job_model = (details.get("model") or {}).get("name", "")
            if model_name and job_model != model_name:
                continue
            state = job_state(job)
            benchmarks = sorted(b.get("id", "?") for b in details.get("benchmarks", []))
            is_success = "SUCCE" in state or "COMPLETE" in state
            is_failure = "FAIL" in state or "ERROR" in state
            if is_success:
                succeeded_jobs += 1
                succeeded_benchmarks.update(b.lower() for b in benchmarks)
            any_failed = any_failed or is_failure
            rows.append(
                {
                    "Job": details.get("name", job.id),
                    "Model": job_model,
                    "Benchmarks": ", ".join(benchmarks),
                    "Status": state,
                }
            )

    if not rows:
        raise ValueError(
            f"No EvalHub jobs found{' for model ' + repr(model_name) if model_name else ''} "
            f"in the last {lookback_jobs} jobs. Is the evaluation pipeline submitting them?"
        )

    coverage_rows = [
        {"Expected Benchmark": b, "Succeeded Run Found": b in succeeded_benchmarks}
        for b in expected
    ]
    passed = (not any_failed) and all(r["Succeeded Run Found"] for r in coverage_rows)

    model_label = model_name or "all models"
    tables = {f"EvalHub jobs ({model_label})": rows}
    if coverage_rows:
        tables[f"Benchmark coverage ({model_label})"] = coverage_rows

    return tables, float(succeeded_jobs), passed
