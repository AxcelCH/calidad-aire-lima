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

from src.config import DATA_CACHE_DIR, SENAMHI_CSV_MIRROR_URL, SENAMHI_CSV_URL

logger = logging.getLogger(__name__)

CACHE_PATTERN = "senamhi_aire_lima*.csv"


def _find_cached_csv() -> Path | None:
    """Devuelve la copia cacheada mas reciente, si existe."""
    candidates = sorted(DATA_CACHE_DIR.glob(CACHE_PATTERN))
    return candidates[-1] if candidates else None


def _download_from(url: str, target: Path, timeout: int) -> None:
    """Descarga `url` a `target` en streaming. Lanza si el servidor responde error."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/octet-stream,*/*",
        "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
        "Referer": "https://www.datosabiertos.gob.pe/dataset/monitoreo-de-los-contaminantes-del-aire-en-lima-metropolitana-servicio-nacional-de",
    }
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def download_csv(url: str = SENAMHI_CSV_URL, timeout: int = 300) -> Path:
    """Descarga el CSV y lo guarda con fecha en el nombre.

    Intenta primero la fuente oficial de SENAMHI; si falla (el portal gob.pe
    bloquea las IPs de proveedores cloud con un 403, p.ej. desde Streamlit
    Cloud), cae al espejo del mismo archivo en un Release de este repo.
    """
    target = DATA_CACHE_DIR / f"senamhi_aire_lima_{dt.date.today():%Y%m%d}.csv"
    logger.info("Descargando CSV de SENAMHI desde %s", url)
    try:
        _download_from(url, target, timeout)
    except requests.RequestException as exc:
        logger.warning("Fuente oficial de SENAMHI fallo (%s); uso el espejo del Release.", exc)
        _download_from(SENAMHI_CSV_MIRROR_URL, target, timeout)
    logger.info("CSV guardado en %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Carga el CSV crudo: usa el cache local o lo descarga si no existe."""
    cached = None if force_download else _find_cached_csv()
    path = cached or download_csv()
    logger.info("Leyendo CSV crudo: %s", path)
    try:  # el CSV oficial es coma-separado; el motor C es ~10x mas rapido
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
    except pd.errors.ParserError:  # por si el portal cambia el separador
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
