"""Cross-cutting utilities shared by ``src.training``, ``src.experiments``,
``src.reporting`` and notebooks.

Currently:
    - :mod:`src.utils.logging`    : unified loguru-frontend logging.
    - :mod:`src.utils.checkpoint` : safe Lightning checkpoint loading.
    - :mod:`src.utils.git`        : capture the current git state for W&B.
"""
