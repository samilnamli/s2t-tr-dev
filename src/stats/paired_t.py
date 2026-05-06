"""Nadeau & Bengio (2003) corrected resampled paired t-test.

The naive paired t-test is anti-conservative under k-fold / repeated
random subsampling cross-validation because the per-fold differences are
correlated (each fold reuses overlapping training samples). Nadeau &
Bengio inflate the variance estimator by ``(1/n + n_test/n_train)`` so
the resulting t-statistic recovers nominal Type-I error.

Reference:
    C. Nadeau, Y. Bengio, "Inference for the Generalization Error",
    Machine Learning 52(3), 2003.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from src.stats.base import BaseStatisticalTest, StatTestResult


class NadeauBengioCorrectedTTest(BaseStatisticalTest):
    """Corrected resampled paired t-test for cross-validated metrics.

    Args:
        alpha: Significance threshold for the ``significant`` flag.
    """

    name = "nadeau_bengio_paired_t"

    def __init__(self, alpha: float = 0.05, **_kwargs: Any):
        super().__init__(alpha=alpha)

    def run(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        model_a: str,
        model_b: str,
        metric: str,
        train_ratio: float,
        test_ratio: float,
        **_kwargs: Any,
    ) -> StatTestResult:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        if train_ratio <= 0 or test_ratio <= 0:
            raise ValueError(
                f"train_ratio and test_ratio must be positive; "
                f"got train={train_ratio}, test={test_ratio}"
            )

        n = int(a.size)
        diffs = a - b
        mean_a = float(a.mean()) if n > 0 else float("nan")
        mean_b = float(b.mean()) if n > 0 else float("nan")
        mean_diff = float(diffs.mean()) if n > 0 else float("nan")
        rho = float(test_ratio) / float(train_ratio)

        if n < 2:
            return StatTestResult(
                test_name=self.name,
                model_a=model_a,
                model_b=model_b,
                n=n,
                metric=metric,
                mean_a=mean_a,
                mean_b=mean_b,
                mean_diff=mean_diff,
                statistic=float("nan"),
                p_value=float("nan"),
                df=float("nan"),
                effect_size=float("nan"),
                significant=False,
                metadata={
                    "train_ratio": float(train_ratio),
                    "test_ratio": float(test_ratio),
                    "rho": rho,
                    "alpha": self.alpha,
                    "note": "n<2; test not computed",
                },
            )

        var_unbiased = float(diffs.var(ddof=1))
        std = math.sqrt(var_unbiased) if var_unbiased > 0 else 0.0
        correction = 1.0 / n + rho
        corrected_var = correction * var_unbiased
        df = float(n - 1)

        note = ""
        if corrected_var <= 0 or not np.isfinite(corrected_var):
            statistic = 0.0
            p_value = 1.0
            note = "degenerate (zero variance across seeds)"
        else:
            statistic = float(mean_diff / math.sqrt(corrected_var))
            p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), df))

        d_z = float(mean_diff / std) if std > 0 else 0.0

        return StatTestResult(
            test_name=self.name,
            model_a=model_a,
            model_b=model_b,
            n=n,
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_diff=mean_diff,
            statistic=statistic,
            p_value=p_value,
            df=df,
            effect_size=d_z,
            significant=bool(p_value < self.alpha and corrected_var > 0),
            metadata={
                "train_ratio": float(train_ratio),
                "test_ratio": float(test_ratio),
                "rho": rho,
                "correction_factor": correction,
                "alpha": self.alpha,
                **({"note": note} if note else {}),
            },
        )
