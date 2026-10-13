# Validate models with ValidMind — OpenShift AI deployment targets.
# See README.md for prerequisites, install, walkthrough, and uninstall.

SHELL := /bin/bash
.DEFAULT_GOAL := help

CHART_DIR ?= ./chart
RELEASE   ?= quickstart
NAMESPACE ?= validmind-quickstart

# Display name of the data science project in the OpenShift AI dashboard.
PROJECT_DISPLAY_NAME ?= ValidMind quickstart

# Source .env for helm/oc recipes (VM_API_HOST, VM_API_KEY, VM_API_SECRET, VM_API_MODEL).
define with_env
	set -a; \
	if [[ -f .env ]]; then source .env; else echo "Warning: .env not found — run: make env-init" >&2; fi; \
	set +a; \
	$(1)
endef

# ValidMind credentials flow into a connection-shaped Secret the workbench
# (and the scheduled pipeline) read as environment variables.
HELM_VM_SETS = \
	--set-string validmind.apiHost="$${VM_API_HOST:-https://api.prod.validmind.ai/api/v1/tracking}" \
	--set-string validmind.apiKey="$${VM_API_KEY}" \
	--set-string validmind.apiSecret="$${VM_API_SECRET}" \
	--set-string validmind.model="$${VM_API_MODEL}"

.PHONY: help env-init env-check env-verify namespace lint template sync-notebook \
	install wait status urls uninstall cleanup-namespace connection-types \
	pipeline logs-pipeline pipeline-compile pipeline-run pipeline-unschedule

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make [target]\n\nTargets:\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2} \
		/^##@/ {printf "\n%s\n", substr($$0, 5)}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Common variables (override on the command line):"
	@echo "  NAMESPACE=$(NAMESPACE)"
	@echo "  RELEASE=$(RELEASE)"

##@ Setup

env-init: ## Copy .env.example to .env
	@test -f .env || cp .env.example .env
	@echo "Created .env — set VM_API_KEY, VM_API_SECRET, and VM_API_MODEL before installing."

env-check: ## Ensure .env exists (creates from .env.example when missing)
	@test -f .env || cp .env.example .env

env-verify: env-check ## Fail if the ValidMind credentials are missing or placeholders
	@$(call with_env, \
	for v in VM_API_KEY VM_API_SECRET VM_API_MODEL; do \
		val="$${!v}"; \
		if [[ -z "$$val" || "$$val" == changeme* ]]; then \
			echo "Set $$v in .env (ValidMind → Model Inventory → your model → Getting Started)"; exit 1; \
		fi; \
	done)
	@echo ".env looks good."

namespace: ## Create or select the data science project
	@oc get project "$(NAMESPACE)" >/dev/null 2>&1 || \
		oc new-project "$(NAMESPACE)" --display-name="$(PROJECT_DISPLAY_NAME)" >/dev/null
	@oc label namespace "$(NAMESPACE)" opendatahub.io/dashboard=true --overwrite >/dev/null
	@oc project "$(NAMESPACE)" >/dev/null
	@echo "Using project $(NAMESPACE)"

##@ Install

sync-notebook: ## Refresh the chart's copies of the notebook and pipeline scripts
	@cp notebooks/validmind-openshift-ai-quickstart.ipynb chart/files/
	@cp pipelines/validmind_pipeline.py pipelines/schedule_pipeline.py chart/files/pipeline/
	@echo "chart/files/ synced from notebooks/ and pipelines/"

install: env-verify namespace sync-notebook ## Deploy the workbench, connection, and notebook
	@$(call with_env,helm upgrade --install "$(RELEASE)" "$(CHART_DIR)" \
		--namespace "$(NAMESPACE)" \
		--create-namespace \
		$(HELM_VM_SETS))
	@$(MAKE) wait
	@$(MAKE) urls

wait: ## Wait until the workbench is ready
	@echo "Waiting for the workbench pod (first start pulls the notebook image — can take a few minutes)..."
	@oc wait --for=jsonpath='{.status.readyReplicas}'=1 statefulset/$(RELEASE)-workbench \
		-n "$(NAMESPACE)" --timeout=900s 2>/dev/null || \
		oc get pods -n "$(NAMESPACE)" -l app=$(RELEASE)-workbench

status: ## Show pods, notebooks, pipelines, and routes in the namespace
	@oc get notebooks,pods,pvc,routes -n "$(NAMESPACE)" 2>/dev/null || oc get pods -n "$(NAMESPACE)"

urls: ## Print the workbench and dashboard URLs
	@WB_HOST=$$(oc get route $(RELEASE)-workbench -n "$(NAMESPACE)" -o jsonpath='{.spec.host}' 2>/dev/null); \
	if [ -z "$$WB_HOST" ]; then \
		WB_HOST=$$(oc get route data-science-gateway -n openshift-ingress -o jsonpath='{.spec.host}' 2>/dev/null); \
	fi; \
	echo "Workbench (direct)     : https://$${WB_HOST:-<pending>}/notebook/$(NAMESPACE)/$(RELEASE)-workbench/lab"
	@DASH_HOST=$$(oc get route -n redhat-ods-applications rhods-dashboard -o jsonpath='{.spec.host}' 2>/dev/null); \
	if [ -z "$$DASH_HOST" ]; then \
		DASH_HOST=$$(oc get route data-science-gateway -n openshift-ingress -o jsonpath='{.spec.host}' 2>/dev/null); \
	fi; \
	echo "OpenShift AI dashboard : https://$${DASH_HOST:-<no dashboard route>}/projects/$(NAMESPACE)"

##@ Connections (cluster admin, one-time)

connection-types: ## Install the ValidMind connection types into the dashboard (cluster admin)
	@oc apply -f connection-types/
	@echo "ValidMind connection types installed — users can now create ValidMind connections in the dashboard."

##@ Scheduled pipeline

pipeline: env-verify sync-notebook ## Enable the scheduled pipeline (deploys pipeline server + recurring run)
	@$(call with_env,helm upgrade "$(RELEASE)" "$(CHART_DIR)" \
		--namespace "$(NAMESPACE)" \
		--reuse-values \
		--set pipeline.enabled=true \
		$(HELM_VM_SETS))
	@echo "Pipeline enabled — follow the setup job with: make logs-pipeline"

logs-pipeline: ## Show logs from the pipeline setup job (includes the on-demand curl recipe)
	@oc logs -f job/$(RELEASE)-pipeline-schedule -n "$(NAMESPACE)"

pipeline-compile: ## Compile the validation pipeline locally to pipelines/dist/
	@cd pipelines && python -m pip install -q -r requirements.txt && python validmind_pipeline.py

pipeline-run: ## Trigger a one-off pipeline run now (local kfp; also see the curl recipe in logs-pipeline)
	@cd pipelines && python schedule_pipeline.py --namespace "$(NAMESPACE)" --run-now

pipeline-unschedule: ## Remove the recurring run
	@cd pipelines && python schedule_pipeline.py --namespace "$(NAMESPACE)" --delete

##@ Development

lint: ## helm lint the chart
	@helm lint "$(CHART_DIR)" \
		--set validmind.apiKey=lint --set validmind.apiSecret=lint --set validmind.model=lint

template: env-check ## Render the chart locally for inspection
	@$(call with_env,helm template "$(RELEASE)" "$(CHART_DIR)" \
		--namespace "$(NAMESPACE)" \
		$(HELM_VM_SETS))

##@ Uninstall

uninstall: ## Remove the Helm release (keeps the namespace and the workbench PVC)
	@helm uninstall "$(RELEASE)" --namespace "$(NAMESPACE)"

cleanup-namespace: ## Delete the entire project (removes ALL data)
	@oc delete project "$(NAMESPACE)"
