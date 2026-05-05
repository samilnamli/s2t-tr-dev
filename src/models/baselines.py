"""ASR routing training-free baselines.

Hierarchy
---------
BaseBaselineConfig          – Hydra structured config (dataclass)
BaseBaseline (ABC)          – tek zorunlu metot: select(batch) -> np.ndarray
    SingleModelBaseline
    OracleBaseline
    RandomBaseline
    WeightedRandomBaseline
    ROVERBaseline
    WeightedROVERBaseline

SelectionMetrics            – wer_stats / selection_accuracy / selection_frequencies
BaselineEvaluator           – DataModule'e bağlanır, tüm baseline'ları çalıştırır

Usage
-----
    dm = ASRDataModule(...)
    evaluator = BaselineEvaluator(datamodule=dm, seed=42)
    results = evaluator.run(save_json="logs/baselines.json")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jiwer
import numpy as np
from loguru import logger

from src.data.dataset import MODEL_NAMES


# ---------------------------------------------------------------------------
# Hydra structured config
# ---------------------------------------------------------------------------

@dataclass
class BaseBaselineConfig:
    """Shared config inherited by every concrete baseline config.

    Hydra requires a ``_target_`` field on concrete subclasses; set it
    to the fully qualified class name (e.g.
    ``src.training.baselines.SingleModelBaseline``).
    """
    model_names: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    seed: int = 42


@dataclass
class SingleModelBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.training.baselines.SingleModelBaseline"
    model_idx: int = 0


@dataclass
class OracleBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.training.baselines.OracleBaseline"


@dataclass
class RandomBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.training.baselines.RandomBaseline"


@dataclass
class WeightedRandomBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.training.baselines.WeightedRandomBaseline"


@dataclass
class ROVERBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.training.baselines.ROVERBaseline"


@dataclass
class WeightedROVERBaselineConfig(BaseBaselineConfig):
    _target_: str = "src.training.baselines.WeightedROVERBaseline"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseBaseline(ABC):
    """Training-free ASR routing baseline.

    Subclasses implement a single method: :meth:`select`, which receives
    a batch dict and returns a ``(N,)`` integer array of selected model
    indices. Everything else (metrics, logging, weight derivation) is
    handled outside.

    Args:
        model_names: Ordered list of ASR model names, matching the WER
            matrix columns in the dataset.
        seed: RNG seed used by stochastic baselines.
    """

    def __init__(self, model_names: list[str] = MODEL_NAMES, seed: int = 42):
        self.model_names = list(model_names)
        self.K = len(model_names)
        self.seed = seed

    @abstractmethod
    def select(self, batch: dict) -> np.ndarray:
        """Select one model per sample.

        Args:
            batch: Dict containing at minimum ``"wer_matrix"`` of shape
                ``(N, K)`` and, for ROVER variants,
                ``"ground_truth"`` and ``"transcription"``.

        Returns:
            Integer array of shape ``(N,)`` with values in ``[0, K)``.
        """

    # Convenience: run over a full list of batch dicts
    def select_all(self, batches: list[dict]) -> np.ndarray:
        return np.concatenate([self.select(b) for b in batches])


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------

class SingleModelBaseline(BaseBaseline):
    """Always selects the same model.

    Args:
        model_idx: Index of the model to always select.
    """

    def __init__(self, model_idx: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.model_idx = model_idx

    def select(self, batch: dict) -> np.ndarray:
        N = batch["wer_matrix"].shape[0]
        return np.full(N, self.model_idx, dtype=np.int64)


class OracleBaseline(BaseBaseline):
    """Selects the model with the lowest WER for each sample (upper bound)."""

    def select(self, batch: dict) -> np.ndarray:
        return batch["wer_matrix"].argmin(axis=-1).astype(np.int64)


class RandomBaseline(BaseBaseline):
    """Selects a uniformly random model for each sample."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(self.seed)

    def select(self, batch: dict) -> np.ndarray:
        N = batch["wer_matrix"].shape[0]
        return self._rng.integers(0, self.K, size=N)


class WeightedRandomBaseline(BaseBaseline):
    """Selects a model by sampling from a learned prior over model quality.

    Weights are derived from training-split WER means: ``w_k ∝ 1 - WER_k``.
    Must call :meth:`fit` with the training WER matrix before :meth:`select`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(self.seed)
        self.weights: Optional[np.ndarray] = None

    def fit(self, wer_train: np.ndarray) -> "WeightedRandomBaseline":
        """Derive selection weights from training-split WER means.

        Args:
            wer_train: ``(N_train, K)`` WER matrix.

        Returns:
            self (for chaining).
        """
        mean = wer_train.mean(axis=0)
        inv = np.maximum(1.0 - mean, 1e-6)
        self.weights = (inv / inv.sum()).astype(np.float32)
        return self

    def select(self, batch: dict) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit(wer_train) before select().")
        N = batch["wer_matrix"].shape[0]
        return self._rng.choice(self.K, size=N, p=self.weights)


class ROVERBaseline(BaseBaseline):
    """ROVER hypothesis combination with uniform system weights.

    Requires ``"ground_truth"`` and ``"transcription"`` keys in the batch.
    Returns the index of the system whose individual hypothesis is closest
    to the ROVER output — used only to unify the interface; the meaningful
    metric is WER of the ROVER transcript itself, stored separately.
    """

    # sentinel for empty alignment slot
    _NULL = ""

    def select(self, batch: dict) -> np.ndarray:
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        return self._rover_select(batch, weights)

    def combine(self, batch: dict) -> list[str]:
        """Return the ROVER-combined transcript for each sample."""
        weights = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        return self._rover_combine(batch, weights)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rover_select(self, batch: dict, weights: np.ndarray) -> np.ndarray:
        """Proxy selection: model whose hypothesis is closest to ROVER output."""
        rover_hyps = self._rover_combine(batch, weights)
        refs = batch["ground_truth"]
        wer_matrix = batch["wer_matrix"]

        # Compute WER between each system and the ROVER output, pick closest
        N = len(refs)
        dist = np.zeros((N, self.K), dtype=np.float32)
        for k, name in enumerate(self.model_names):
            sys_hyps = batch["transcription"][name]
            for i, (rover_h, sys_h) in enumerate(zip(rover_hyps, sys_hyps)):
                r = rover_h if rover_h.strip() else "<empty>"
                h = sys_h if sys_h.strip() else ""
                dist[i, k] = float(jiwer.wer(r, h))

        return dist.argmin(axis=-1).astype(np.int64)

    def _rover_combine(self, batch: dict, weights: np.ndarray) -> list[str]:
        N = len(batch["ground_truth"])
        return [
            self._rover_one(
                [batch["transcription"][n][i] for n in self.model_names], weights
            )
            for i in range(N)
        ]

    def _rover_one(self, hypotheses: list[str], weights: np.ndarray) -> str:
        token_lists = [h.split() for h in hypotheses]
        cn = self._build_confusion_network(token_lists)
        return " ".join(self._vote(cn, weights)).strip()

    @classmethod
    def _build_confusion_network(cls, hypotheses: list[list[str]]) -> list[list[str]]:
        if not hypotheses:
            return []
        cn: list[list[str]] = [[tok] for tok in hypotheses[0]]
        for h in hypotheses[1:]:
            cn = cls._pairwise_align(cn, h)
        return cn

    @classmethod
    def _pairwise_align(
        cls, a: list[list[str]], b: list[str]
    ) -> list[list[str]]:
        NULL = cls._NULL
        n, m = len(a), len(b)
        n_sys = len(a[0]) if a else 0
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(m + 1):
            dp[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub = 0 if b[j - 1] in a[i - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j - 1] + sub,
                    dp[i - 1, j] + 1,
                    dp[i, j - 1] + 1,
                )
        out: list[list[str]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub = 0 if b[j - 1] in a[i - 1] else 1
                if dp[i, j] == dp[i - 1, j - 1] + sub:
                    out.append(a[i - 1] + [b[j - 1]])
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
                out.append(a[i - 1] + [NULL])
                i -= 1
                continue
            out.append([NULL] * n_sys + [b[j - 1]])
            j -= 1
        out.reverse()
        return out

    @classmethod
    def _vote(cls, cn: list[list[str]], weights: np.ndarray) -> list[str]:
        NULL = cls._NULL
        out: list[str] = []
        for slot in cn:
            scores: dict[str, float] = {}
            for tok, w in zip(slot, weights):
                scores[tok] = scores.get(tok, 0.0) + float(w)
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            if best != NULL:
                out.append(best)
        return out


class WeightedROVERBaseline(ROVERBaseline):
    """ROVER with per-system weights derived from training-split WER means.

    Must call :meth:`fit` before :meth:`select`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights: Optional[np.ndarray] = None

    def fit(self, wer_train: np.ndarray) -> "WeightedROVERBaseline":
        """Derive ROVER weights from training-split WER means.

        Args:
            wer_train: ``(N_train, K)`` WER matrix.

        Returns:
            self (for chaining).
        """
        mean = wer_train.mean(axis=0)
        inv = np.maximum(1.0 - mean, 1e-6)
        self.weights = (inv / inv.sum()).astype(np.float32)
        return self

    def select(self, batch: dict) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit(wer_train) before select().")
        return self._rover_select(batch, self.weights)

    def combine(self, batch: dict) -> list[str]:
        if self.weights is None:
            raise RuntimeError("Call fit(wer_train) before combine().")
        return self._rover_combine(batch, self.weights)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class SelectionMetrics:
    """Compute routing evaluation metrics from selection indices and WER matrix.

    All methods are stateless; pass arrays directly.
    """

    @staticmethod
    def wer_stats(
        selected_idx: np.ndarray,
        wer_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Mean and SEM of per-sample WER for the selected models.

        Args:
            selected_idx: ``(N,)`` integer array of selected model indices.
            wer_matrix: ``(N, K)`` float array of per-sample per-model WER.

        Returns:
            Dict with ``wer_mean``, ``wer_sem``, ``n``.
        """
        N = wer_matrix.shape[0]
        selected_wer = wer_matrix[np.arange(N), selected_idx]
        sem = float(selected_wer.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0
        return {
            "wer_mean": float(selected_wer.mean()),
            "wer_sem": sem,
            "n": int(N),
        }

    @staticmethod
    def selection_accuracy(
        selected_idx: np.ndarray,
        wer_matrix: np.ndarray,
    ) -> dict[str, float]:
        """Fraction of samples where the selected model is oracle-optimal.

        Args:
            selected_idx: ``(N,)`` integer array of selected model indices.
            wer_matrix: ``(N, K)`` float array.

        Returns:
            Dict with ``selection_accuracy`` and ``n``.
        """
        oracle_idx = wer_matrix.argmin(axis=-1)
        acc = float(np.mean(selected_idx == oracle_idx))
        return {"selection_accuracy": acc, "n": int(len(selected_idx))}

    @staticmethod
    def selection_frequencies(
        selected_idx: np.ndarray,
        model_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """How often each model is selected.

        Args:
            selected_idx: ``(N,)`` integer array of selected model indices.
            model_names: Ordered list of model names.

        Returns:
            Dict mapping model name to ``{"count": int, "freq": float}``.
        """
        K = len(model_names)
        counts = np.bincount(selected_idx, minlength=K)
        total = len(selected_idx)
        return {
            name: {"count": int(c), "freq": float(c / total)}
            for name, c in zip(model_names, counts)
        }

    @classmethod
    def compute_all(
        cls,
        selected_idx: np.ndarray,
        wer_matrix: np.ndarray,
        model_names: list[str],
    ) -> dict:
        """Convenience wrapper: compute all metrics at once.

        Returns:
            Dict with keys ``wer_mean``, ``wer_sem``, ``selection_accuracy``,
            ``selection_frequencies``, ``n``.
        """
        result = {}
        result.update(cls.wer_stats(selected_idx, wer_matrix))
        result.update(cls.selection_accuracy(selected_idx, wer_matrix))
        result["selection_frequencies"] = cls.selection_frequencies(
            selected_idx, model_names
        )
        return result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """Run all baselines over a DataModule's test split.

    Uses the same train/test split as the trained models since it reads
    directly from the DataModule. Baselines requiring a training prior
    (WeightedRandom, WeightedROVER) are fit on the training split.

    Args:
        datamodule: Any Lightning DataModule whose datasets expose
            ``wer_matrix``, ``ground_truth``, and ``transcription``
            per sample.
        model_names: Ordered list of ASR model names.
        seed: RNG seed forwarded to stochastic baselines.
    """

    def __init__(
        self,
        datamodule,
        model_names: list[str] = MODEL_NAMES,
        seed: int = 42,
    ):
        self.datamodule = datamodule
        self.model_names = list(model_names)
        self.seed = seed
        self.metrics = SelectionMetrics()

        self._baselines: dict[str, BaseBaseline] = {
            **{
                f"single_{name}": SingleModelBaseline(
                    model_idx=k, model_names=model_names, seed=seed
                )
                for k, name in enumerate(model_names)
            },
            "oracle":          OracleBaseline(model_names=model_names, seed=seed),
            "random":          RandomBaseline(model_names=model_names, seed=seed),
            "weighted_random": WeightedRandomBaseline(model_names=model_names, seed=seed),
            "rover":           ROVERBaseline(model_names=model_names, seed=seed),
            "weighted_rover":  WeightedROVERBaseline(model_names=model_names, seed=seed),
        }

    def run(self, save_json: Optional[str] = None) -> dict:
        """Evaluate all baselines and return a results dict.

        Args:
            save_json: Optional path to write results as JSON.

        Returns:
            Nested dict: ``{baseline_name: {wer_mean, wer_sem,
            selection_accuracy, selection_frequencies, n}}``.
        """
        self.datamodule.setup(stage="fit")
        self.datamodule.setup(stage="test")

        train_batches = self._collect_batches(self.datamodule.train_dataset)
        test_batches  = self._collect_batches(self.datamodule.test_dataset)

        wer_train = np.concatenate([b["wer_matrix"] for b in train_batches])

        # Fit train-prior baselines
        self._baselines["weighted_random"].fit(wer_train)
        self._baselines["weighted_rover"].fit(wer_train)

        results: dict[str, dict] = {}
        for name, baseline in self._baselines.items():
            selected_idx = baseline.select_all(test_batches)
            wer_test = np.concatenate([b["wer_matrix"] for b in test_batches])
            results[name] = SelectionMetrics.compute_all(
                selected_idx, wer_test, self.model_names
            )

        self._log(results)

        if save_json:
            self._save(results, save_json)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_batches(dataset) -> list[dict]:
        """Wrap each dataset sample in a length-1 pseudo-batch."""
        batches = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # wer_matrix: ensure 2-D (1, K)
            wer = np.atleast_2d(sample["wer_matrix"])
            batch: dict = {
                "wer_matrix": wer,
                "ground_truth": [sample["ground_truth"]],
                "transcription": {
                    n: [sample["transcription"][n]]
                    for n in sample["transcription"]
                },
            }
            batches.append(batch)
        return batches

    def _log(self, results: dict):
        logger.info("=== Baseline Evaluation Results ===")
        for name, stats in results.items():
            logger.info(
                "  {:<24} wer={:.4f} ± {:.4f}  sel_acc={:.4f}  n={}",
                name,
                stats["wer_mean"],
                stats["wer_sem"],
                stats["selection_accuracy"],
                stats["n"],
            )
            logger.info(
                "  {:<24} selection_freq: {}",
                "",
                {k: f"{v['freq']:.2%}" for k, v in stats["selection_frequencies"].items()},
            )

    @staticmethod
    def _save(results: dict, path: str):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved baseline results to {}", path)