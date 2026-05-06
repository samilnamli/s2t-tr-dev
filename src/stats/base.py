"""Base abstractions for statistical tests over per-seed metric arrays.

Inputs are always two paired numpy arrays of shape ``(n_seeds,)``: one per
method being compared. Outputs are :class:`StatTestResult` dataclasses,
flat and trivially serializable so that orchestration code can stack
them into a pandas DataFrame and log it as an MLflow artifact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict

import numpy as np


@dataclass(frozen=True)
class StatTestResult:
    """One pairwise comparison result.

    ``metadata`` carries test-specific fields (correction factor, ratios,
    diagnostic notes). It is flattened into the row produced by
    :meth:`to_row` so that downstream consumers don't need to know the
    test class.
    """

    test_name: str
    model_a: str
    model_b: str
    n: int
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float
    statistic: float
    p_value: float
    df: float
    effect_size: float
    significant: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        meta = row.pop("metadata", {})
        for k, v in meta.items():
            row.setdefault(k, v)
        return row


class BaseStatisticalTest(ABC):
    """Common contract for paired-sample statistical tests.

    Subclasses implement :meth:`run` and set the class-level ``name``.
    The interface is deliberately narrow: tests do not know about MLflow,
    DataFrames, or method-name bookkeeping — those are the experiment
    harness's concern.
    """

    name: ClassVar[str]
    requires_paired: ClassVar[bool] = True

    def __init__(self, alpha: float = 0.05, **_kwargs: Any):
        self.alpha = float(alpha)

    @abstractmethod
    def run(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        model_a: str,
        model_b: str,
        metric: str,
        **kwargs: Any,
    ) -> StatTestResult:
        """Compare two paired arrays of shape ``(n,)``."""

    def summary(self, result: StatTestResult) -> str:
        verdict = "≠" if result.significant else "≈"
        return (
            f"{result.test_name}: {result.model_a} {verdict} {result.model_b}  "
            f"(Δ={result.mean_diff:+.4f}, t={result.statistic:.3f}, "
            f"df={result.df:.1f}, p={result.p_value:.4g})"
        )
