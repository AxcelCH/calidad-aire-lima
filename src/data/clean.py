"""Limpieza y preparacion del CSV historico de SENAMHI.

Pipeline: normalizar columnas -> parsear fecha/hora -> convertir a numerico ->
filtrar nulos (solo para entrenamiento) -> agregar a promedio diario por estacion.
"""
from __future__ import annotations

import logging
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

# Nombres candidatos (ya normalizados) -> nombre canonico
COLUMN_ALIASES: dict[str, str] = {
    "ESTACION": "estacion",
    "NOMBRE_ESTACION": "estacion",
    "FECHA": "fecha",
    "HORA": "hora",
    "PM10": "pm10",
    "PM2_5": "pm25",
    "PM25": "pm25",
    "NO2": "no2",
    "LONGITUD": "longitud",
    "LATITUD": "latitud",
    "ALTITUD": "altitud",
    "DISTRITO": "distrito",
}

POLLUTANTS = ["pm10", "pm25", "no2"]


def _normalize(name: str) -> str:
    """Quita tildes/espacios/puntos y pasa a MAYUSCULAS: 'PM2.5 ' -> 'PM2_5'."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return text.strip().upper().replace(" ", "_").replace(".", "_").replace(",", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas del CSV crudo a nombres canonicos en minuscula."""
    renamed = {col: COLUMN_ALIASES.get(_normalize(col), _normalize(col).lower()) for col in df.columns}
    return df.rename(columns=renamed)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la columna 'fecha_hora' (datetime) y 'fecha_dia' (date).

    Soporta FECHA como YYYYMMDD (formato del portal) o como fecha ya parseable,
    y HORA como '13:00', 13 o 130000 (HHMMSS, formato del CSV oficial).
    """
    out = df.copy()
    fecha_raw = out["fecha"].astype(str).str.strip().str.split(".").str[0]
    fecha = pd.to_datetime(fecha_raw, format="%Y%m%d", errors="coerce")
    if fecha.isna().mean() > 0.5:  # el portal cambio de formato: intento generico
        fecha = pd.to_datetime(fecha_raw, errors="coerce", dayfirst=True)

    hora_raw = out.get("hora")
    horas = pd.Series(0, index=out.index)
    if hora_raw is not None:
        hora_str = hora_raw.astype(str).str.strip().str.split(".").str[0]
        hora_num = pd.to_numeric(hora_str.str.replace(":", ""), errors="coerce").fillna(0)
        # El CSV oficial usa HHMMSS (ej. 50000 = 05:00:00, 230000 = 23:00:00);
        # si el valor es <=23 ya viene como hora simple
        horas = hora_num.where(hora_num <= 23, hora_num // 10000)
        horas = horas.clip(0, 23).astype(int)

    out["fecha_hora"] = fecha + pd.to_timedelta(horas, unit="h")
    out["fecha_dia"] = fecha.dt.date
    return out


def coerce_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los contaminantes a numerico (los vacios quedan como NaN)."""
    out = df.copy()
    for col in POLLUTANTS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            # Valores negativos son errores de sensor -> NaN
            out.loc[out[col] < 0, col] = pd.NA
            out[col] = out[col].astype(float)
    return out


def clean_hourly(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia el CSV crudo horario. Devuelve (df_limpio, reporte_calidad).

    El reporte documenta el % de registros descartados (regla de negocio 2:
    se descartan del entrenamiento pero se conservan en el crudo).
    """
    df = normalize_columns(df_raw)
    df = parse_datetime(df)
    df = coerce_pollutants(df)

    total = len(df)
    valid_date = df["fecha_hora"].notna()
    any_pollutant = df[[c for c in POLLUTANTS if c in df.columns]].notna().any(axis=1)
    mask = valid_date & any_pollutant
    cleaned = df.loc[mask].copy()
    # 'CAMPO_DE_MARTE' -> 'Campo De Marte' (coincide con el mapeo de estaciones WAQI)
    cleaned["estacion"] = (
        cleaned["estacion"].astype(str).str.replace("_", " ").str.strip().str.title()
    )

    report = {
        "registros_totales": total,
        "registros_validos": int(mask.sum()),
        "pct_descartado": round(100 * (1 - mask.sum() / max(total, 1)), 2),
        "estaciones": sorted(cleaned["estacion"].unique().tolist()),
        "fecha_min": str(cleaned["fecha_dia"].min()),
        "fecha_max": str(cleaned["fecha_dia"].max()),
        "nulos_por_contaminante": {
            c: int(cleaned[c].isna().sum()) for c in POLLUTANTS if c in cleaned.columns
        },
    }
    logger.info("Limpieza: %s", report)
    return cleaned, report


def aggregate_daily(df_hourly: pd.DataFrame, min_hours: int = 12) -> pd.DataFrame:
    """Promedio diario por estacion (regla de negocio 3: el ECA es promedio 24h).

    Solo se conservan dias con al menos `min_hours` mediciones de PM2.5,
    para que el promedio diario sea representativo de las 24 horas.
    """
    pollutant_cols = [c for c in POLLUTANTS if c in df_hourly.columns]
    grouped = (
        df_hourly.groupby(["estacion", "fecha_dia"])
        .agg(**{c: (c, "mean") for c in pollutant_cols}, horas_pm25=("pm25", "count"))
        .reset_index()
    )
    grouped = grouped[grouped["horas_pm25"] >= min_hours].copy()
    grouped["fecha_dia"] = pd.to_datetime(grouped["fecha_dia"])
    return grouped.sort_values(["estacion", "fecha_dia"]).reset_index(drop=True)
