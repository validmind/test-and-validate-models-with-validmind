"""ValidMind tests backed by a TrustyAI service on Red Hat OpenShift AI.

Importing this package registers every test under the ``trustyai.`` namespace
(via the ``@vm.test`` decorator), so after ``import validmind_trustyai`` they
can be run like any built-in test::

    vm.tests.run_test("trustyai.StatisticalParityDifference", params={...})
"""

from .DisparateImpactRatio import DisparateImpactRatio
from .EvalHubBenchmarkScores import EvalHubBenchmarkScores
from .EvalHubRunCoverage import EvalHubRunCoverage
from .FourierMMDDrift import FourierMMDDrift
from .GroupFairnessComparison import GroupFairnessComparison
from .InferenceDataHealth import InferenceDataHealth
from .KSTestDrift import KSTestDrift
from .MeanShiftDrift import MeanShiftDrift
from .PredictionMeanMonitor import PredictionMeanMonitor
from .ScheduledMetricCoverage import ScheduledMetricCoverage
from .SchemaIntegrity import SchemaIntegrity
from .StatisticalParityDifference import StatisticalParityDifference

__all__ = [
    "DisparateImpactRatio",
    "EvalHubBenchmarkScores",
    "EvalHubRunCoverage",
    "FourierMMDDrift",
    "GroupFairnessComparison",
    "InferenceDataHealth",
    "KSTestDrift",
    "MeanShiftDrift",
    "PredictionMeanMonitor",
    "ScheduledMetricCoverage",
    "SchemaIntegrity",
    "StatisticalParityDifference",
]
