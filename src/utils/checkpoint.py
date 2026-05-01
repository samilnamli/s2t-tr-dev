"""Safe Lightning checkpoint loading with the legacy ``weights_only`` semantics.

PyTorch 2.6 hardened :func:`torch.load` to default to ``weights_only=True``,
which rejects ``pickle`` payloads that aren't whitelisted. Lightning's
.ckpt files contain a small amount of non-tensor metadata (callback
states, hyperparameters) that trips this guard.

Because we only ever load checkpoints we wrote ourselves, the
hardening adds no security and just breaks loading. The previous
implementation handled this by **monkey-patching the global
``torch.load``** at module import time, which leaks process-wide
behavior into anything that imports ``src.training.train`` or
``src.training.visualize``.

This module replaces that with two safe alternatives:

* :func:`safe_torch_load` — a drop-in wrapper that calls
  ``torch.load(..., weights_only=False)`` exactly once.
* :func:`legacy_torch_load` — a context manager that swaps
  ``torch.load`` for the legacy variant only inside its block, and
  restores the original on exit (even on exception). Useful when a
  third-party loader (e.g. ``LightningModule.load_from_checkpoint``)
  calls ``torch.load`` itself and we cannot pass kwargs through.

Usage:
    >>> from src.utils.checkpoint import legacy_torch_load
    >>> with legacy_torch_load():
    ...     module = ASRSelectorModule.load_from_checkpoint(path)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


def safe_torch_load(path: str, **kwargs: Any) -> Any:
    """Call :func:`torch.load` with ``weights_only=False``.

    Args:
        path: Path to the checkpoint file.
        **kwargs: Forwarded to :func:`torch.load`. Any explicit
            ``weights_only`` is preserved (caller wins).
    """
    kwargs.setdefault("weights_only", False)
    return torch.load(path, **kwargs)


@contextmanager
def legacy_torch_load() -> Iterator[None]:
    """Locally swap :func:`torch.load` for the legacy ``weights_only=False`` variant.

    Restores the original on exit, even if the wrapped block raises.
    Safe to nest; each entry stacks an extra restoration frame.

    Implementation detail.
        We **force** ``weights_only=False`` rather than ``setdefault``-ing it.
        Lightning >= 2.6 explicitly passes ``weights_only=True`` from
        ``trainer.test(ckpt_path=...)`` down to ``cloud_io._load``, which
        means a ``setdefault`` is a silent no-op and the legacy fallback
        never kicks in. Inside this context the caller has explicitly
        opted into loose-mode loading, so we honor that intent over any
        wrapped library's default.
    """
    original = torch.load

    def _patched(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]
