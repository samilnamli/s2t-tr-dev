"""tests/test_baselines.py

Lightweight tests for src/training/baselines.py.
Reads only the first N_SAMPLES rows from the parquet to stay within RAM limits.
Does NOT require a real DataModule — uses a minimal stub.
"""

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.data.dataset import MODEL_NAMES, WER_COLUMNS
from src.training.baselines import (
    BaselineEvaluator,
    OracleBaseline,
    RandomBaseline,
    ROVERBaseline,
    SelectionMetrics,
    SingleModelBaseline,
    WeightedRandomBaseline,
    WeightedROVERBaseline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARQUET_PATH = "data/processed/edinburghcstr_ami/combined_features.parquet"
N_SAMPLES    = 200   # only first N rows — keeps RAM low
K            = len(MODEL_NAMES)

TRANSCRIPTION_COLUMNS = {
    "hubert":   "hubert_transcription",
    "whisper":  "whisper_transcription",
    "wav2vec2": "w2v2_transcription",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_parquet():
    """Load a tiny slice of the parquet. Returns (wer, transcriptions, refs)."""
    wer_cols   = [WER_COLUMNS[n] for n in MODEL_NAMES]
    trans_cols = [TRANSCRIPTION_COLUMNS[n] for n in MODEL_NAMES]
    meta_cols  = ["ground_truth"]

    pf = pq.ParquetFile(PARQUET_PATH)
    # read only first batch that covers N_SAMPLES rows
    batch = next(pf.iter_batches(batch_size=N_SAMPLES, columns=wer_cols + trans_cols + meta_cols))
    table = batch  # pyarrow RecordBatch supports column access like a Table

    wer = np.stack(
        [table.column(c).to_pylist() for c in wer_cols], axis=-1
    ).astype(np.float32)

    transcriptions = {n: [str(s) for s in table.column(TRANSCRIPTION_COLUMNS[n]).to_pylist()]
                      for n in MODEL_NAMES}
    refs = [str(s) for s in table.column("ground_truth").to_pylist()]

    return wer, transcriptions, refs


@pytest.fixture(scope="module")
def batches(small_parquet):
    """Convert parquet slice into pseudo-batch list consumed by baselines."""
    wer, transcriptions, refs = small_parquet
    N = wer.shape[0]
    return [
        {
            "wer_matrix":   wer[i : i + 1],          # (1, K)
            "ground_truth": [refs[i]],
            "transcription": {n: [transcriptions[n][i]] for n in MODEL_NAMES},
        }
        for i in range(N)
    ]


@pytest.fixture(scope="module")
def wer_matrix(small_parquet):
    return small_parquet[0]   # (N, K)


@pytest.fixture(scope="module")
def train_wer(wer_matrix):
    """Use first 80 % of the slice as a proxy train split."""
    n_train = int(len(wer_matrix) * 0.8)
    return wer_matrix[:n_train]


# ---------------------------------------------------------------------------
# BaseBaseline — shared contract tests
# ---------------------------------------------------------------------------

class TestSelectContract:
    """Every baseline must return a (N,) int array in [0, K)."""

    def _check(self, baseline, batches, wer_matrix):
        idx = baseline.select_all(batches)
        assert idx.shape == (len(wer_matrix),), "wrong output shape"
        assert idx.dtype in (np.int32, np.int64), "must be integer dtype"
        assert idx.min() >= 0 and idx.max() < K, "index out of range"

    def test_single_model(self, batches, wer_matrix):
        self._check(SingleModelBaseline(model_idx=0), batches, wer_matrix)

    def test_oracle(self, batches, wer_matrix):
        self._check(OracleBaseline(), batches, wer_matrix)

    def test_random(self, batches, wer_matrix):
        self._check(RandomBaseline(), batches, wer_matrix)

    def test_weighted_random(self, batches, wer_matrix, train_wer):
        b = WeightedRandomBaseline().fit(train_wer)
        self._check(b, batches, wer_matrix)

    def test_rover(self, batches, wer_matrix):
        self._check(ROVERBaseline(), batches, wer_matrix)

    def test_weighted_rover(self, batches, wer_matrix, train_wer):
        b = WeightedROVERBaseline().fit(train_wer)
        self._check(b, batches, wer_matrix)


# ---------------------------------------------------------------------------
# Baseline semantics
# ---------------------------------------------------------------------------

class TestBaselineSemantics:

    def test_single_model_always_same(self, batches):
        for k in range(K):
            b = SingleModelBaseline(model_idx=k)
            idx = b.select_all(batches)
            assert (idx == k).all(), f"SingleModelBaseline(model_idx={k}) must always return {k}"

    def test_oracle_is_lower_bound(self, batches, wer_matrix):
        """Oracle WER must be ≤ every other baseline's WER."""
        oracle_idx = OracleBaseline().select_all(batches)
        oracle_wer = wer_matrix[np.arange(len(wer_matrix)), oracle_idx].mean()

        for k in range(K):
            single_wer = wer_matrix[:, k].mean()
            assert oracle_wer <= single_wer + 1e-6, (
                f"Oracle WER ({oracle_wer:.4f}) > single model {MODEL_NAMES[k]} "
                f"WER ({single_wer:.4f})"
            )

    def test_weighted_random_weights_sum_to_one(self, train_wer):
        b = WeightedRandomBaseline().fit(train_wer)
        assert abs(b.weights.sum() - 1.0) < 1e-5

    def test_weighted_rover_weights_sum_to_one(self, train_wer):
        b = WeightedROVERBaseline().fit(train_wer)
        assert abs(b.weights.sum() - 1.0) < 1e-5

    def test_weighted_random_requires_fit(self, batches):
        with pytest.raises(RuntimeError):
            WeightedRandomBaseline().select_all(batches)

    def test_weighted_rover_requires_fit(self, batches):
        with pytest.raises(RuntimeError):
            WeightedROVERBaseline().select_all(batches)


# ---------------------------------------------------------------------------
# SelectionMetrics
# ---------------------------------------------------------------------------

class TestSelectionMetrics:

    def test_wer_stats_shape(self, wer_matrix):
        idx = np.zeros(len(wer_matrix), dtype=np.int64)
        stats = SelectionMetrics.wer_stats(idx, wer_matrix)
        assert "wer_mean" in stats and "wer_sem" in stats and "n" in stats
        assert stats["n"] == len(wer_matrix)
        assert stats["wer_mean"] >= 0.0

    def test_selection_accuracy_oracle_is_one(self, wer_matrix):
        oracle_idx = wer_matrix.argmin(axis=-1)
        result = SelectionMetrics.selection_accuracy(oracle_idx, wer_matrix)
        assert abs(result["selection_accuracy"] - 1.0) < 1e-6, (
            "Oracle selection accuracy must be 1.0"
        )

    def test_selection_frequencies_sum_to_one(self, wer_matrix):
        idx = wer_matrix.argmin(axis=-1)
        freqs = SelectionMetrics.selection_frequencies(idx, MODEL_NAMES)
        total_freq = sum(v["freq"] for v in freqs.values())
        assert abs(total_freq - 1.0) < 1e-6

    def test_compute_all_keys(self, wer_matrix):
        idx = np.zeros(len(wer_matrix), dtype=np.int64)
        result = SelectionMetrics.compute_all(idx, wer_matrix, MODEL_NAMES)
        for key in ("wer_mean", "wer_sem", "selection_accuracy", "selection_frequencies", "n"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# DataModule stub — BaselineEvaluator integration smoke test
# ---------------------------------------------------------------------------

class _StubDataset:
    """Minimal dataset wrapping the parquet slice."""

    def __init__(self, wer, transcriptions, refs):
        self._wer   = wer
        self._trans = transcriptions
        self._refs  = refs

    def __len__(self):
        return len(self._wer)

    def __getitem__(self, i):
        return {
            "wer_matrix":    self._wer[i],
            "ground_truth":  self._refs[i],
            "transcription": {n: self._trans[n][i] for n in MODEL_NAMES},
        }


class _StubDataModule:
    def __init__(self, wer, transcriptions, refs):
        n_train = int(len(wer) * 0.8)
        self.train_dataset = _StubDataset(wer[:n_train], 
                                          {n: transcriptions[n][:n_train] for n in MODEL_NAMES},
                                          refs[:n_train])
        self.test_dataset  = _StubDataset(wer[n_train:],
                                          {n: transcriptions[n][n_train:] for n in MODEL_NAMES},
                                          refs[n_train:])

    def setup(self, stage=None):
        pass   # already set up in __init__


class TestBaselineEvaluatorSmoke:

    def test_run_returns_all_baselines(self, small_parquet):
        wer, transcriptions, refs = small_parquet
        dm = _StubDataModule(wer, transcriptions, refs)
        evaluator = BaselineEvaluator(datamodule=dm)
        results = evaluator.run()

        expected_keys = (
            [f"single_{n}" for n in MODEL_NAMES]
            + ["oracle", "random", "weighted_random", "rover", "weighted_rover"]
        )
        for key in expected_keys:
            assert key in results, f"Missing baseline in results: {key}"

    def test_run_result_values_are_valid(self, small_parquet):
        wer, transcriptions, refs = small_parquet
        dm = _StubDataModule(wer, transcriptions, refs)
        results = BaselineEvaluator(datamodule=dm).run()

        for name, stats in results.items():
            assert 0.0 <= stats["wer_mean"], f"{name}: negative wer_mean"
            assert stats["wer_sem"] >= 0.0,  f"{name}: negative wer_sem"
            assert 0.0 <= stats["selection_accuracy"] <= 1.0, \
                f"{name}: selection_accuracy out of [0,1]"