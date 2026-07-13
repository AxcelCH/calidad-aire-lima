"""Ingesta del CSV historico oficial de SENAMHI (datosabiertos.gob.pe).

Regla de ingesta: se descarga una vez y se cachea en data_cache/ con la fecha
en el nombre, para no depender de que el portal este arriba el dia de la demo.
Si ya existe una copia local (descargada a mano o por una corrida anterior),
se usa esa sin tocar la red.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_CACHE_DIR, SENAMHI_CSV_URL

logger = logging.getLogger(__name__)

CACHE_PATTERN = "senamhi_aire_lima*.csv"


def _find_cached_csv() -> Path | None:
    """Devuelve la copia cacheada mas reciente, si existe."""
    candidates = sorted(DATA_CACHE_DIR.glob(CACHE_PATTERN))
    return candidates[-1] if candidates else None


def download_csv(url: str = SENAMHI_CSV_URL, timeout: int = 300) -> Path:
    """Descarga el CSV oficial y lo guarda con fecha en el nombre."""
    target = DATA_CACHE_DIR / f"senamhi_aire_lima_{dt.date.today():%Y%m%d}.csv"
    logger.info("Descargando CSV de SENAMHI desde %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (proyecto academico UNMSM - mineria de datos)"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    logger.info("CSV guardado en %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Carga el CSV crudo: usa el cache local o lo descarga si no existe."""
    cached = None if force_download else _find_cached_csv()
    path = cached or download_csv()
    logger.info("Leyendo CSV crudo: %s", path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
