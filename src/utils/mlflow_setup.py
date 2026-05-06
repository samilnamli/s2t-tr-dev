"""MLflow + DagsHub authentication and tracking-URI bootstrap.

Resolution order for the tracking URI:
  1. ``cfg.tracking_uri`` (explicit override from CLI / YAML)
  2. ``$DAGSHUB_TRACKING_URI`` env var
  3. ``$MLFLOW_TRACKING_URI`` env var
  4. ``./mlruns`` local fallback

DagsHub auth: if ``$DAGSHUB_USER_TOKEN`` is set, both
``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD`` are set
to the token (DagsHub's token-based MLflow auth scheme).
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger
import mlflow
from omegaconf import DictConfig


def _resolve_tracking_uri(cfg_uri: Optional[str]) -> str:
    if cfg_uri:
        return cfg_uri
    return (
        os.environ.get("DAGSHUB_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_URI") or "mlruns"
    )


def _install_dagshub_auth() -> None:
    token = os.environ.get("DAGSHUB_USER_TOKEN")
    if not token:
        return
    os.environ["MLFLOW_TRACKING_USERNAME"] = (
        os.environ.get("DAGSHUB_USERNAME", os.environ.get("MLFLOW_TRACKING_USERNAME", ""))
        or "token"
    )
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token


def setup_mlflow(cfg: DictConfig) -> str:
    """Configure mlflow client and return the resolved tracking URI.

    Args:
        cfg: A DictConfig with ``tracking_uri`` (optional) and
            ``experiment_name`` (required).

    Returns:
        Resolved tracking URI (after env-var fallbacks).
    """
    _install_dagshub_auth()
    tracking_uri = _resolve_tracking_uri(cfg.get("tracking_uri"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    logger.info(
        "mlflow: tracking_uri={} experiment={}",
        tracking_uri,
        cfg.experiment_name,
    )
    return tracking_uri
