"""Guardado y carga de modelos entrenados (joblib) en models_cache/."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from src.config import MODELS_CACHE_DIR

logger = logging.getLogger(__name__)


def save_model(obj: Any, name: str) -> Path:
    """Serializa un modelo/objeto con joblib. Devuelve la ruta escrita."""
    path = MODELS_CACHE_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    logger.info("Modelo guardado: %s", path)
    return path


def load_model(name: str) -> Any | None:
    """Carga un modelo guardado, o None si no existe."""
    path = MODELS_CACHE_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
