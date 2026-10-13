"""validmind-trustyai — ValidMind tests that read live bias, drift, and
data-health evidence from a TrustyAI service on Red Hat OpenShift AI.

Usage::

    import validmind as vm
    import validmind_trustyai  # registers the trustyai.* tests

    vm.init()  # VM_API_HOST / VM_API_KEY / VM_API_SECRET / VM_API_MODEL

    vm.tests.run_test(
        "trustyai.StatisticalParityDifference",
        params={
            "model_id": "loan-model",
            "protected_attribute": "gender",
            "privileged_group": 1,
            "unprivileged_group": 0,
            "outcome_name": "approved",
            "favorable_outcome": 1,
        },
    ).log()

The TrustyAI service location and credentials come from ``TRUSTYAI_URL`` /
``TRUSTYAI_TOKEN`` (with in-cluster ServiceAccount defaults) or per-test
``trustyai_url`` / ``trustyai_token`` params.
"""

from .client import TrustyAIClient, TrustyAIError, metric_values
from . import tests

__version__ = "0.1.0"

#: Every test id this package registers.
TEST_IDS = [f"trustyai.{name}" for name in tests.__all__]

__all__ = ["TrustyAIClient", "TrustyAIError", "metric_values", "tests", "TEST_IDS"]
