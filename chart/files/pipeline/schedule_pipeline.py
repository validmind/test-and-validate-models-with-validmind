#!/usr/bin/env python3
"""Upload the ValidMind validation pipeline and manage its schedule.

Talks to the data science pipelines (Kubeflow Pipelines v2) API in an
OpenShift AI project. Works both:

- locally    — resolves the ds-pipeline route with `oc` and authenticates
               with your `oc whoami -t` token
- in-cluster — uses the pod's ServiceAccount token against the
               ds-pipeline-<dspa> service (the Helm hook job does this)

Actions:
    (default)     upload/refresh the pipeline and ensure the recurring run
    --run-now     also trigger a one-off run immediately
    --delete      remove the recurring run(s) created by this script

On success, prints a ready-to-use `curl` command for triggering the pipeline
on demand over plain HTTP.
"""

import argparse
import os
import subprocess
import sys
import time

PIPELINE_NAME = "validmind-validation"
EXPERIMENT_NAME = "validmind-quickstart"
RECURRING_RUN_NAME = "validmind-validation-scheduled"

SA_TOKEN = "/var/run/secrets/kubernetes.io/serviceaccount/token"
SA_CA = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
SA_K8S_CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def resolve_host(args):
    if args.host:
        return args.host
    if os.environ.get("KFP_HOST"):
        return os.environ["KFP_HOST"]
    host = sh("oc", "get", "route", f"ds-pipeline-{args.dspa}", "-n", args.namespace,
              "-o", "jsonpath={.spec.host}")
    if host:
        return f"https://{host}"
    sys.exit(
        "Could not resolve the pipeline API host. Pass --host, set KFP_HOST, or "
        f"ensure route ds-pipeline-{args.dspa} exists in {args.namespace}."
    )


def resolve_token(args):
    if args.token:
        return args.token
    if os.environ.get("KFP_TOKEN"):
        return os.environ["KFP_TOKEN"]
    if os.path.exists(SA_TOKEN):
        with open(SA_TOKEN) as f:
            return f.read().strip()
    token = sh("oc", "whoami", "-t")
    if token:
        return token
    sys.exit("No API token. Pass --token, set KFP_TOKEN, or log in with oc.")


def external_host(args, fallback):
    """Best externally-reachable pipeline API URL, for the printed curl recipe.

    Locally the resolved route already is external. In-cluster, look the
    route up through the Kubernetes API with the pod's ServiceAccount.
    """
    if "svc" not in fallback:
        return fallback
    host = sh("oc", "get", "route", f"ds-pipeline-{args.dspa}", "-n", args.namespace,
              "-o", "jsonpath={.spec.host}")
    if not host and os.path.exists(SA_TOKEN):
        try:
            import requests

            with open(SA_TOKEN) as f:
                token = f.read().strip()
            resp = requests.get(
                "https://kubernetes.default.svc/apis/route.openshift.io/v1"
                f"/namespaces/{args.namespace}/routes/ds-pipeline-{args.dspa}",
                headers={"Authorization": f"Bearer {token}"},
                verify=SA_K8S_CA if os.path.exists(SA_K8S_CA) else False,
                timeout=10,
            )
            if resp.ok:
                host = resp.json().get("spec", {}).get("host", "")
        except Exception:
            pass
    return f"https://{host}" if host else fallback


def existing_recurring_runs(client, experiment_id):
    """kfp 2.x names the field recurring_runs; some builds expose recurringruns."""
    resp = client.list_recurring_runs(experiment_id=experiment_id)
    return getattr(resp, "recurring_runs", None) or getattr(resp, "recurringruns", None) or []


def compiled_package():
    here = os.path.dirname(os.path.abspath(__file__))
    package = os.path.join(here, "dist", "validmind_pipeline.yaml")
    if not os.path.exists(package):
        print("Compiling pipeline...")
        subprocess.run([sys.executable, os.path.join(here, "validmind_pipeline.py")], check=True)
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dspa", default=os.environ.get("DSPA_NAME", "dspa"))
    parser.add_argument("--host", help="Pipeline API base URL (default: resolve the route)")
    parser.add_argument("--token", help="Bearer token (default: SA token or oc whoami -t)")
    parser.add_argument("--cron", default=os.environ.get("PIPELINE_CRON", "0 0 2 * * *"),
                        help="6-field cron, seconds first (default: daily 02:00 UTC)")
    parser.add_argument("--run-now", action="store_true", help="Trigger a one-off run now")
    parser.add_argument("--delete", action="store_true", help="Remove the recurring run")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification")
    args = parser.parse_args()

    import kfp

    host, token = resolve_host(args), resolve_token(args)
    ssl_ca_cert = SA_CA if (os.path.exists(SA_CA) and "svc" in host) else None
    client = kfp.Client(
        host=host,
        existing_token=token,
        ssl_ca_cert=ssl_ca_cert,
        verify_ssl=not args.insecure,
    )

    experiment = client.create_experiment(EXPERIMENT_NAME, namespace=args.namespace)
    experiment_id = getattr(experiment, "experiment_id", None) or experiment.id

    # ------------------------------------------------------------- delete
    if args.delete:
        removed = 0
        for rr in existing_recurring_runs(client, experiment_id):
            if rr.display_name == RECURRING_RUN_NAME:
                client.delete_recurring_run(rr.recurring_run_id)
                removed += 1
        print(f"Removed {removed} recurring run(s) named {RECURRING_RUN_NAME}.")
        return

    # ------------------------------------------------- upload / new version
    package = compiled_package()
    pipeline_id = client.get_pipeline_id(PIPELINE_NAME)
    if pipeline_id is None:
        pipeline = client.upload_pipeline(package, pipeline_name=PIPELINE_NAME,
                                          namespace=args.namespace)
        pipeline_id = pipeline.pipeline_id
        version_id = None  # initial version was created with the pipeline
    else:
        version = client.upload_pipeline_version(
            package,
            pipeline_version_name=f"{PIPELINE_NAME}-{time.strftime('%Y%m%d-%H%M%S')}",
            pipeline_id=pipeline_id,
        )
        version_id = version.pipeline_version_id
    if version_id is None:
        versions = client.list_pipeline_versions(pipeline_id, sort_by="created_at desc")
        version_id = versions.pipeline_versions[0].pipeline_version_id
    print(f"Pipeline {PIPELINE_NAME}: id={pipeline_id} version={version_id}")

    # ------------------------------------------------------ recurring run
    for rr in existing_recurring_runs(client, experiment_id):
        if rr.display_name == RECURRING_RUN_NAME:
            client.delete_recurring_run(rr.recurring_run_id)
    recurring = client.create_recurring_run(
        experiment_id=experiment_id,
        job_name=RECURRING_RUN_NAME,
        cron_expression=args.cron,
        pipeline_id=pipeline_id,
        version_id=version_id,
        max_concurrency=1,
        no_catchup=True,
        enabled=True,
    )
    print(f"Recurring run scheduled (cron: {args.cron}): {recurring.recurring_run_id}")

    # ------------------------------------------------------------ run now
    if args.run_now:
        run = client.run_pipeline(
            experiment_id=experiment_id,
            job_name=f"{PIPELINE_NAME}-manual-{time.strftime('%Y%m%d-%H%M%S')}",
            pipeline_id=pipeline_id,
            version_id=version_id,
        )
        print(f"Triggered run: {getattr(run, 'run_id', run)}")

    # ------------------------------------- on-demand HTTP trigger recipe
    trigger_host = external_host(args, host)
    print(
        "\nTrigger this pipeline on demand with a plain HTTP call:\n\n"
        f"  curl -sk -X POST '{trigger_host}/apis/v2beta1/runs' \\\n"
        "    -H \"Authorization: Bearer $(oc whoami -t)\" \\\n"
        "    -H 'Content-Type: application/json' \\\n"
        "    -d '{\n"
        f'      "display_name": "validmind-validation-on-demand",\n'
        f'      "experiment_id": "{experiment_id}",\n'
        f'      "pipeline_version_reference": {{\n'
        f'        "pipeline_id": "{pipeline_id}",\n'
        f'        "pipeline_version_id": "{version_id}"\n'
        "      }\n"
        "    }'\n\n"
        "Any bearer token with access to the project works (a ServiceAccount "
        "token for automation, e.g. from a CI webhook)."
    )


if __name__ == "__main__":
    main()
