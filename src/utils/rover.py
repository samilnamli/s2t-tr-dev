"""ROVER hypothesis combination — confusion-network construction and voting.

Pure functions used by ROVERBaseline / WeightedROVERBaseline.
"""

from __future__ import annotations

import numpy as np

_NULL = ""


def rover_combine(
    transcriptions: dict[str, list[str]],
    model_names: list[str],
    weights: np.ndarray,
) -> list[str]:
    """Return the ROVER-combined transcript for each sample in the batch.

    Args:
        transcriptions: Maps model name to list of N hypotheses.
        model_names: Ordered list of system names defining the column order.
        weights: ``(K,)`` per-system weights summing to 1.
    """
    N = len(transcriptions[model_names[0]])
    return [_rover_one([transcriptions[n][i] for n in model_names], weights) for i in range(N)]


def _rover_one(hypotheses: list[str], weights: np.ndarray) -> str:
    token_lists = [h.split() for h in hypotheses]
    cn = _build_confusion_network(token_lists)
    return " ".join(_vote(cn, weights)).strip()


def _build_confusion_network(hypotheses: list[list[str]]) -> list[list[str]]:
    if not hypotheses:
        return []
    cn: list[list[str]] = [[tok] for tok in hypotheses[0]]
    for h in hypotheses[1:]:
        cn = _pairwise_align(cn, h)
    return cn


def _pairwise_align(a: list[list[str]], b: list[str]) -> list[list[str]]:
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
            out.append(a[i - 1] + [_NULL])
            i -= 1
            continue
        out.append([_NULL] * n_sys + [b[j - 1]])
        j -= 1
    out.reverse()
    return out


def _vote(cn: list[list[str]], weights: np.ndarray) -> list[str]:
    out: list[str] = []
    for slot in cn:
        scores: dict[str, float] = {}
        for tok, w in zip(slot, weights):
            scores[tok] = scores.get(tok, 0.0) + float(w)
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        if best != _NULL:
            out.append(best)
    return out
