# Test and validate models with ValidMind

Independent model validation on OpenShift AI: a workbench pre-loaded with a
ValidMind quickstart notebook, a reusable ValidMind connection type for the
dashboard, and a scheduled pipeline that re-runs the validation tests and
keeps your model documentation current — including tests that read live bias
and drift metrics from TrustyAI.

## Table of Contents

- [Overview](#overview)
- [Detailed description](#detailed-description)
  - [Architecture diagrams](#architecture-diagrams)
- [Requirements](#requirements)
  - [Minimum hardware requirements](#minimum-hardware-requirements)
  - [Minimum software requirements](#minimum-software-requirements)
  - [Required user permissions](#required-user-permissions)
- [Deploy](#deploy)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Validating the deployment](#validating-the-deployment)
  - [Walkthrough](#walkthrough)
  - [The scheduled pipeline](#the-scheduled-pipeline)
  - [Triggering validation on demand over HTTP](#triggering-validation-on-demand-over-http)
  - [The ValidMind connection type (cluster admins)](#the-validmind-connection-type-cluster-admins)
  - [TrustyAI-backed tests](#trustyai-backed-tests)
  - [Delete](#delete)
- [Repository structure](#repository-structure)
- [References](#references)
- [Technical details](#technical-details)
- [Tags](#tags)

## Overview

[ValidMind](https://validmind.com) is a model risk management platform: model
inventory, documentation, independent validation, and ongoing monitoring,
built for regulated industries. This quickstart connects it to Red Hat
OpenShift AI. You get a **workbench** with a tested notebook that trains a
reference model, runs the ValidMind test suite, and uploads the evidence to
your model's documentation; a **connection type** so ValidMind credentials
work like any other OpenShift AI connection; and a **data science pipeline**
that re-runs validation on a schedule or on demand.

## Detailed description

Model documentation goes stale the moment it's written. A model validated in
January and redeployed in March is, from a governance standpoint, an
undocumented model — unless re-validation is cheap enough to run continuously.

This quickstart makes re-validation cheap on OpenShift AI, in three layers:

1. **Interactive** (`make install`): a JupyterLab workbench appears in your
   data science project, pre-loaded with a notebook that installs the
   ValidMind Library, loads data from an S3 connection (or a synthetic
   fallback), trains a scikit-learn classifier, runs individual tests and the
   full classifier test suite, and logs everything to the ValidMind Platform.
   The ValidMind credentials arrive as a *connection* attached to the
   workbench — `vm.init()` needs no arguments, and no secrets live in the
   notebook.

2. **Scheduled** (`make pipeline`): the same validation steps, packaged as a
   Kubeflow (data science) pipeline. A Helm hook deploys a pipeline server if
   the project doesn't have one, uploads the pipeline, and creates a recurring
   run (daily 02:00 UTC by default). The pipeline consumes the *same*
   connection Secret as the workbench, and the KFP REST API gives you a plain
   `curl` one-liner to trigger a validation run from CI, a webhook, or a
   model-deployment hook.

3. **Production evidence** (`validmind-trustyai`): twelve ValidMind tests that
   don't recompute anything — they read live fairness (SPD, DIR), drift
   (MeanShift, KS, FourierMMD), output-rate, data-health, and
   monitoring-coverage metrics from the TrustyAI service running next to your
   deployed model, plus EvalHub benchmark evidence (are the promised
   evaluations running, and do the latest scores in MLflow meet their bars),
   apply pass/fail policy, and log it all to ValidMind. TrustyAI collects;
   ValidMind documents and governs.

### Architecture diagrams

![Architecture diagram: the ValidMind connection secret feeds both a JupyterLab workbench (pre-loaded with the quickstart notebook by an init container) and a scheduled data science pipeline; both run ValidMind tests and send results to the ValidMind Platform; the optional validmind-trustyai package reads bias and drift metrics from a TrustyAI service beside the deployed model](docs/images/architecture.svg)

One connection, three consumers: the workbench notebook for interactive
validation, the scheduled pipeline for continuous validation, and (optionally)
the TrustyAI-backed tests for production monitoring evidence. All three log to
the same model in the ValidMind Platform.

## Requirements

### Minimum hardware requirements

No GPU is required.

- **Workbench:** 1 vCPU request / 2 vCPU limit, 4 GiB / 8 GiB memory, 5 GiB PV
- **Pipeline (optional):** the pipeline server pods (~1 vCPU / 2 GiB across
  them) plus a 10 GiB PV for artifact storage, and a short-lived validation
  pod per run

### Minimum software requirements

- Red Hat OpenShift AI 2.16 or later (self-managed or Cloud Service), with
  workbenches enabled; data science pipelines enabled for the scheduled
  pipeline; TrustyAI enabled for the TrustyAI-backed tests
- `oc` CLI 4.14+, `helm` 3.12+, `make`
- A ValidMind account with a registered model — you need the API key, API
  secret, and model identifier from **Model Inventory → your model →
  Getting Started**

### Required user permissions

Any user who can create a project (or be given one) and deploy applications
via Helm: Notebooks (kubeflow.org), PVCs, Secrets, ConfigMaps, Jobs,
ServiceAccounts, RoleBindings, and (for the pipeline)
DataSciencePipelinesApplications.

Cluster admin is needed **only** for the optional one-time
`make connection-types` (it writes ConfigMaps into `redhat-ods-applications`).

## Deploy

### Prerequisites

- `oc` logged in to a cluster with OpenShift AI
- `helm` and `make` installed
- Your ValidMind credentials (API key, secret, model identifier)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rh-ai-quickstart/test-and-validate-models-with-validmind.git
cd test-and-validate-models-with-validmind
```

2. Configure your ValidMind credentials:
```bash
make env-init          # creates .env
# edit .env: set VM_API_KEY, VM_API_SECRET, VM_API_MODEL
```

3. Install (creates the project, deploys the workbench + connection, seeds the
notebook, prints the URLs):
```bash
make install
```

Defaults: project `validmind-quickstart`, release `quickstart` — override like
`make install NAMESPACE=my-project`.

<details>
<summary>Prefer plain Helm? (what the Makefile runs)</summary>

```bash
PROJECT="validmind-quickstart"
oc new-project ${PROJECT}
oc label namespace ${PROJECT} opendatahub.io/dashboard=true
helm install quickstart ./chart --namespace ${PROJECT} \
  --set validmind.apiKey=YOUR_API_KEY \
  --set validmind.apiSecret=YOUR_API_SECRET \
  --set validmind.model=YOUR_MODEL_CUID
```

</details>

### Validating the deployment

1. Check the workbench is running:
```bash
make status        # or: oc get notebooks,pods,routes -n validmind-quickstart
```
Expected: pod `quickstart-workbench-0` `2/2 Running` (notebook + oauth proxy).

2. Get the URLs:
```bash
make urls
```

### Walkthrough

1. Open the workbench URL (or find the project in the OpenShift AI dashboard
   and open the **ValidMind workbench**) — log in with your OpenShift user.
2. Open `validmind-openshift-ai-quickstart.ipynb`, seeded into the home
   directory.
3. Run the notebook top to bottom: it installs the ValidMind Library, calls
   `vm.init()` (no arguments — the connection provides the credentials),
   loads data from an attached S3 connection or falls back to a synthetic
   dataset, trains a gradient-boosting classifier, runs five individual tests
   and then the full `classifier_full_suite`.
4. Open the ValidMind Platform → **Model Inventory** → your model →
   **Documentation** and watch the test results populate the corresponding
   sections.

### The scheduled pipeline

```bash
make pipeline          # enables pipeline.enabled=true on the release
make logs-pipeline     # watch the setup job; it prints the on-demand curl recipe
```

This deploys a pipeline server (DSPA + a small MinIO for artifacts) unless the
project already has one (`--set pipeline.deployServer=false` in that case),
uploads the `validmind-validation` pipeline, and creates a recurring run —
default cron `0 0 2 * * *` (6 fields, seconds first: daily 02:00 UTC), override
with `--set pipeline.cron=...`.

The pipeline runs the same steps as the notebook, headlessly (LLM description
generation off), reading the same ValidMind connection Secret. Point it at
real training data by naming an S3 connection Secret:
`--set pipeline.s3DataSecret=my-s3-connection`.

### Triggering validation on demand over HTTP

The pipeline server's KFP REST API is the ergonomic HTTP trigger. Once the
pipeline is scheduled, one POST creates a run:

```bash
curl -sk -X POST "https://$(oc get route ds-pipeline-dspa -n validmind-quickstart -o jsonpath='{.spec.host}')/apis/v2beta1/runs" \
  -H "Authorization: Bearer $(oc whoami -t)" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "validmind-validation-on-demand",
    "experiment_id": "<EXPERIMENT_ID>",
    "pipeline_version_reference": {
      "pipeline_id": "<PIPELINE_ID>",
      "pipeline_version_id": "<VERSION_ID>"
    }
  }'
```

`make logs-pipeline` prints this command with all three ids filled in. Any
bearer token with access to the project works — use a ServiceAccount token for
automation (CI job, deployment webhook, ValidMind workflow step):

```bash
oc create token quickstart-pipeline-setup -n validmind-quickstart --duration=24h
```

There is no equivalent for the *notebook*: workbenches are interactive pods,
and driving the Jupyter kernel API remotely is neither supported nor
reproducible. The pipeline is the on-demand execution surface — same code,
proper run history, parameters, and retries. (`make pipeline-run` wraps the
same API call locally via the `kfp` SDK.)

### The ValidMind connection type (cluster admins)

```bash
make connection-types
```

Installs two *connection types* into the OpenShift AI dashboard (one-time,
cluster admin, works on RHOAI 2.16+):

- **ValidMind** — API host, API key, API secret (masked), model identifier.
  Creating one produces a Secret whose keys are exactly the environment
  variables the ValidMind Library reads (`VM_API_HOST`, `VM_API_KEY`,
  `VM_API_SECRET`, `VM_API_MODEL`), so *any* workbench, pipeline, or model
  server it's attached to gets zero-boilerplate `vm.init()`.
- **ValidMind (OIDC sign-in)** — issuer/client-id/audience only, no secrets
  stored in the cluster; `vm.init()` runs the OAuth device flow in the
  browser. Recommended where long-lived credentials in namespace Secrets are
  not acceptable.

The quickstart itself doesn't require this step — `make install` creates a
connection-shaped Secret directly — but with the type installed, users can
mint additional ValidMind connections from the dashboard UI for any project,
and rotate credentials there.

Notes for the security-conscious:

- A connection is an Opaque Secret in the project namespace: base64, not
  encryption. Anyone with `get secrets` in the namespace can read it. Enable
  etcd encryption, restrict RBAC, or use the OIDC variant.
- Rotating a key requires restarting attached workbenches (env vars are set
  at pod start).
- External Secrets Operator / Vault can own the credential: any Secret carrying
  the same labels/annotations (`opendatahub.io/dashboard: "true"`,
  `opendatahub.io/connection-type-ref: validmind`) appears as a ValidMind
  connection in the dashboard.

### TrustyAI-backed tests

The [`validmind-trustyai`](validmind-trustyai/) package adds twelve ValidMind
tests that read live evidence from the TrustyAI stack — group fairness (SPD,
DIR, multi-group sweep), drift (MeanShift, KS, FourierMMD), output-rate
monitoring, inference-data health, schema integrity, scheduled-metric
coverage, and two EvalHub tests (evaluation-run coverage via the EvalHub API,
benchmark scores vs thresholds via MLflow). See its
[README](validmind-trustyai/README.md) for the test-by-test rationale,
prerequisites (TrustyAI enabled, payload logging wired, reference data tagged;
EvalHub deployed for the `EvalHub*` pair), and usage from both workbenches and
pipelines.

### Delete

1. Remove the Helm release (keeps the namespace and the workbench PVC, so
   notebook edits survive a re-install):
```bash
make uninstall
```

2. Delete the whole project including data:
```bash
make cleanup-namespace     # or: oc delete project validmind-quickstart
```

## Repository structure

```
.
├── Makefile                     # make install / pipeline / connection-types / status / uninstall
├── .env.example                 # VM_API_HOST, VM_API_KEY, VM_API_SECRET, VM_API_MODEL
├── quickstart.yaml              # Publication metadata
├── notebooks/                   # The quickstart notebook (canonical copy)
├── chart/                       # Helm chart
│   ├── values.yaml              # validmind.*, workbench.*, pipeline.*
│   ├── templates/               # Notebook CR, PVC, connection Secret, notebook ConfigMap
│   ├── templates/pipeline/      # MinIO, DSPA, scripts ConfigMap, schedule hook Job
│   └── files/                   # Chart copies of the notebook + pipeline scripts (make sync-notebook)
├── connection-types/            # ValidMind connection types for the dashboard (cluster admin)
├── pipelines/                   # KFP v2 pipeline definition + schedule/trigger script
├── validmind-trustyai/          # pip package: 10 TrustyAI-backed ValidMind tests
└── docs/images/                 # Architecture diagram
```

## References

- [ValidMind documentation](https://docs.validmind.ai)
- [ValidMind Library](https://github.com/validmind/validmind-library)
- [TrustyAI](https://trustyai.org)
- [Red Hat OpenShift AI: working with connections](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/)
- [Kubeflow Pipelines v2 SDK](https://www.kubeflow.org/docs/components/pipelines/)

## Technical details

- **Workbench**: a minimal `kubeflow.org/v1` Notebook CR; the OpenShift AI
  notebook controller injects the oauth proxy, ServiceAccount, TLS, and Route
  (both `inject-oauth` and `inject-auth` annotations are set for 2.x/3.x
  compatibility). An init container seeds the notebook into the PVC-backed
  home directory on first start and never overwrites user edits.
- **Credentials**: one connection-shaped Secret (labels/annotations the
  dashboard recognizes) feeds the workbench via `envFrom` and the pipeline
  task via `kfp-kubernetes` `use_secret_as_env`. The ValidMind Library reads
  `VM_API_HOST` / `VM_API_KEY` / `VM_API_SECRET` / `VM_API_MODEL` natively, so
  `vm.init()` takes no arguments anywhere in the quickstart.
- **Notebook install cell**: pins `--index-url https://pypi.org/simple` (some
  workbench images configure an internal index without `validmind`) and
  `plotly<6` / `kaleido==0.2.1` (tested against the 2025.1 Standard Data
  Science image).
- **Pipeline**: KFP v2, one component (UBI9 Python 3.12 base,
  `packages_to_install=["validmind", "jinja2", "boto3"]` — `jinja2` is
  imported by the validmind library but not declared as its dependency as of
  2.13.x, so a clean container needs it explicitly), caching disabled — a
  validation run must actually run. The component pins one asyncio event loop
  for the whole process: the validmind API client caches an aiohttp session
  across calls, which breaks under the fresh-loop-per-call behavior of plain
  (non-Jupyter) Python. Headless init:
  `generate_descriptions=False` skips LLM description calls. The recurring run
  is created through the KFP API by a Helm post-install/post-upgrade hook Job
  (in-cluster, ServiceAccount token against `ds-pipeline-<dspa>:8443`);
  re-running `make pipeline` is idempotent (replaces the schedule).
- **Pipeline server**: deployed only when missing —
  `DataSciencePipelinesApplication` named `dspa` with `externalStorage`
  pointing at a chart-managed FS-mode MinIO (the same image the pipelines
  operator uses for samples). One DSPA per project is an OpenShift AI limit.
- **TrustyAI tests**: use only endpoints stable across the Java (RHOAI 2.x)
  and Python (RHOAI 3.x) TrustyAI services; auth is any bearer token with
  `view` on the namespace (ServiceAccount token in pipelines, `oc whoami -t`
  interactively). Scalar-returning tests double as ValidMind unit metrics, so
  scheduled runs accumulate over-time monitoring series.

## Tags

**Title:** Test and validate models with ValidMind
**Description:** Independent model validation on OpenShift AI: a pre-loaded workbench, a ValidMind connection type, and a scheduled pipeline that keeps model documentation current — with TrustyAI-backed bias and drift evidence.
**Industry:** Financial Services
**Product:** Red Hat OpenShift AI
**Use case:** Model risk management, validation, governance
**Partner:** ValidMind
**Contributor org:** ValidMind
