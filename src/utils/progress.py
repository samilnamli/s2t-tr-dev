"""Lightning progress-bar selection that works in both notebooks and TTYs.

Colab's stderr proxy mishandles ``\\r`` once any non-CR-terminated line
sneaks between two ``tqdm`` refreshes — and the unified loguru sink in
:mod:`src.utils.logging` regularly emits such lines from Lightning's
stdlib loggers. The combination produces one new line per training
step instead of an in-place progress bar.

Strategy:
    * In a Jupyter / Colab front-end → return a :class:`RichProgressBar`,
      which renders via ``rich`` (independent of ``\\r`` semantics).
    * In a TTY → return :class:`TQDMProgressBar` from ``tqdm.auto`` (the
      loguru sink in :mod:`src.utils.logging` is already ``tqdm.write``-
      aware so the two coexist cleanly).
    * If ``rich`` is unavailable for any reason → fall back to TQDM so
      training never fails because of cosmetic logging issues.
"""

from __future__ import annotations

import pytorch_lightning as pl


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    # ZMQInteractiveShell = jupyter / colab; TerminalInteractiveShell = ipython tty.
    return ip.__class__.__name__ == "ZMQInteractiveShell"


def make_progress_bar(refresh_rate: int = 50) -> pl.callbacks.Callback:
    """Return a Lightning progress-bar callback appropriate for the env."""
    if _is_notebook():
        try:
            return pl.callbacks.RichProgressBar(refresh_rate=refresh_rate)
        except (ImportError, ModuleNotFoundError):
            pass
    return pl.callbacks.TQDMProgressBar(refresh_rate=refresh_rate)
