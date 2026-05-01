"""Unified loguru-frontend logging for the whole pipeline.

Why this file exists.
    The project mixes stdlib :mod:`logging` (used by ``src/training``,
    Lightning, transformers, datasets, hydra, wandb, …), :mod:`loguru`
    (used by ``src/data/get_processed.py``), and bare ``print`` calls
    (used by some 3rd-party progress bars). Without intervention each
    of these renders with its own format, level filter, and stream,
    which makes a single ``make`` run unreadable on Colab and on
    headless container terminals.

What this module does.
    :func:`setup_unified_logging` configures loguru as the single sink
    and installs an :class:`InterceptHandler` on the stdlib root logger
    so that every record emitted via stdlib ``logging.getLogger(...)``
    is forwarded to loguru with the correct level and depth (the
    original caller's frame). After calling this once at process
    start, every component speaks loguru's format with consistent
    timestamps, levels, and ANSI colors on a TTY.

Usage.
    >>> from src.utils.logging import setup_unified_logging
    >>> setup_unified_logging(level="INFO")
    >>> import logging
    >>> logging.getLogger("transformers").info("hello")  # routed to loguru
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Iterable, Optional, Union

from loguru import logger

# Logger-name prefixes that are very noisy at INFO and rarely useful for
# our work. We coerce them (and every descendant) down to WARNING so the
# terminal doesn't drown in chatter but records are still captured for
# a debug session that explicitly raises the level.
_NOISY_LOGGER_PREFIXES: tuple[str, ...] = (
    "datasets",
    "transformers",
    "huggingface_hub",
    "wandb",
    "matplotlib",
    "PIL",
    "urllib3",
    "filelock",
    "fsspec",
    "asyncio",
    # torch internals dump cache stats / fake tensor traces at INFO and
    # are noise for our work. We silence the specific subloggers known
    # to chatter, not bare "torch", so genuinely useful torch warnings
    # (cudnn, autograd anomalies) still surface.
    "torch._subclasses",
    "torch._inductor",
    "torch._dynamo",
)


def _is_noisy(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


class InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records into loguru.

    Mirrors the recipe from the loguru docs, with two refinements:
      - We resolve the original caller's frame so source-file/line
        information in loguru's output points at the actual call site,
        not at the stdlib ``logging`` machinery.
      - We respect the stdlib record's level via :func:`Logger.level`,
        falling back to the numeric level when no name is registered
        (e.g. for custom levels emitted by 3rd-party libraries).
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - I/O
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "<level>{message}</level>"
)


def setup_unified_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    *,
    fmt: str = _DEFAULT_FORMAT,
    enqueue: bool = False,
    extra_silenced: Iterable[str] = (),
) -> None:
    """Install loguru as the sole sink for the whole process.

    Args:
        level: Minimum level emitted to the terminal.
        log_file: Optional path to also tee log records into. The file
            sink is rotation-free; rotate externally if needed.
        fmt: Loguru format string. The default is ANSI-colored.
        enqueue: Pass through to loguru's sinks. Useful when multiple
            processes share a sink; harmless otherwise.
        extra_silenced: Extra logger names to coerce down to WARNING.
    """
    logger.remove()
    logger.add(sys.stderr, level=level, format=fmt, enqueue=enqueue, backtrace=False)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format=fmt,
            enqueue=enqueue,
            backtrace=False,
            encoding="utf-8",
        )

    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(level)

    noisy_prefixes = (*_NOISY_LOGGER_PREFIXES, *tuple(extra_silenced))

    # Pre-create the parent loggers so we can pin their levels even when
    # the underlying library hasn't been imported yet. This makes the
    # filtering deterministic regardless of import order.
    for prefix in noisy_prefixes:
        nl = logging.getLogger(prefix)
        nl.handlers = []
        nl.setLevel(logging.WARNING)
        nl.propagate = True

    # Then walk every existing logger and decide.
    for name in list(logging.root.manager.loggerDict.keys()):
        existing = logging.getLogger(name)
        existing.handlers = []
        existing.propagate = True
        if _is_noisy(existing.name, noisy_prefixes):
            existing.setLevel(logging.WARNING)
        else:
            existing.setLevel(logging.NOTSET)
