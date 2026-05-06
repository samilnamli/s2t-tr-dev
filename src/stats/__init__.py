"""Statistical tests for routing-method comparisons.

Hierarchy
---------
BaseStatisticalTest (ABC)
    NadeauBengioCorrectedTTest   – corrected paired t-test for k-fold CV

Tests are run at the parent-experiment level, after all (method, seed)
runs have finished, on paired per-seed metric arrays. Results are logged
as ``results/pairwise_stat_tests.{json,md}`` artifacts on the parent run.
"""

from src.stats.base import BaseStatisticalTest, StatTestResult
from src.stats.paired_t import NadeauBengioCorrectedTTest

__all__ = ["BaseStatisticalTest", "StatTestResult", "NadeauBengioCorrectedTTest"]
