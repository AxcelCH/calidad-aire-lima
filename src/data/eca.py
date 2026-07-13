"""Reglas de negocio: Estandares de Calidad Ambiental (ECA) para aire.

Umbrales segun D.S. N. 003-2017-MINAM (promedio de 24 horas):
- PM2.5: 50 ug/m3
- PM10 : 100 ug/m3

Funciones puras, sin dependencias externas: son la base de los tests.
"""
from __future__ import annotations

import pandas as pd

ECA_PM25_24H: float = 50.0
ECA_PM10_24H: float = 100.0


def exceeds_pm25(daily_mean: float) -> bool:
    """True si el promedio diario de PM2.5 excede el ECA (> 50 ug/m3)."""
    return daily_mean > ECA_PM25_24H


def exceeds_pm10(daily_mean: float) -> bool:
    """True si el promedio diario de PM10 excede el ECA (> 100 ug/m3)."""
    return daily_mean > ECA_PM10_24H


def label_exceedance(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas binarias de excedencia ECA a un DataFrame diario.

    Espera columnas 'pm25' y/o 'pm10' con promedios de 24h.
    Devuelve una copia con 'excede_pm25' y/o 'excede_pm10' (0/1).
    """
    out = df_daily.copy()
    if "pm25" in out.columns:
        out["excede_pm25"] = (out["pm25"] > ECA_PM25_24H).astype(int)
    if "pm10" in out.columns:
        out["excede_pm10"] = (out["pm10"] > ECA_PM10_24H).astype(int)
    return out
