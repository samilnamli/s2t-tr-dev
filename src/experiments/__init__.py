"""Experiment package.

Hydra config is loaded from YAML under ``configs/``; structured
dataclasses and ConfigStore registration have been intentionally
removed. ``BaseExperiment`` (in :mod:`src.experiments.base`) is the
single generic runner — subclassing is not required.
"""

from src.experiments.base import BaseExperiment

__all__ = ["BaseExperiment"]
