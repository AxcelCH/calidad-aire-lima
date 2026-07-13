"""Cliente de la API WAQI (api.waqi.info) para el dato en vivo del Panel 4.

Se llama solo bajo demanda (nunca en background) para cuidar la cuota gratuita.
La API devuelve el AQI (indice EPA de EE.UU.), no ug/m3; se incluye la
conversion inversa oficial (breakpoints EPA 2024) para comparar contra el ECA.
"""
from __future__ import annotations

import logging

import requests

from src.config import WAQI_API_TOKEN

logger = logging.getLogger(__name__)

# Estaciones activas en Lima verificadas el 13/07/2026 (uid de WAQI)
STATIONS_WAQI: dict[str, int] = {
    "San Borja": 379,
    "Campo De Marte": 380,
    "Santa Anita": 381,
    "Villa Maria Del Triunfo": 382,
    "San Juan De Lurigancho": 7577,
    "Carabayllo": 7579,
    "San Martin De Porres": 7580,
    "Puente Piedra": 7581,
}

# Breakpoints EPA (mayo 2024) para PM2.5: (aqi_lo, aqi_hi, conc_lo, conc_hi)
_PM25_BREAKPOINTS = [
    (0, 50, 0.0, 9.0),
    (51, 100, 9.1, 35.4),
    (101, 150, 35.5, 55.4),
    (151, 200, 55.5, 125.4),
    (201, 300, 125.5, 225.4),
    (301, 500, 225.5, 325.4),
]


def aqi_to_pm25_concentration(aqi: float) -> float | None:
    """Convierte AQI (EPA) de PM2.5 a concentracion aproximada en ug/m3."""
    for aqi_lo, aqi_hi, c_lo, c_hi in _PM25_BREAKPOINTS:
        if aqi_lo <= aqi <= aqi_hi:
            return round(c_lo + (aqi - aqi_lo) * (c_hi - c_lo) / (aqi_hi - aqi_lo), 1)
    return None


def get_live_reading(station_name: str, timeout: int = 10) -> dict | None:
    """Consulta el valor en vivo de una estacion. Devuelve None si falla.

    Nunca lanza excepcion hacia la UI (regla de degradacion controlada):
    el Panel 4 debe seguir funcionando aunque la API no responda.
    """
    uid = STATIONS_WAQI.get(station_name)
    if uid is None or not WAQI_API_TOKEN:
        logger.warning("Estacion sin uid WAQI o token ausente: %s", station_name)
        return None
    try:
        resp = requests.get(
            f"https://api.waqi.info/feed/@{uid}/",
            params={"token": WAQI_API_TOKEN},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "ok":
            logger.warning("WAQI status != ok: %s", payload)
            return None
        data = payload["data"]
        iaqi = data.get("iaqi", {})
        aqi_pm25 = iaqi.get("pm25", {}).get("v")
        return {
            "aqi_general": data.get("aqi"),
            "aqi_pm25": aqi_pm25,
            "pm25_estimado_ugm3": aqi_to_pm25_concentration(aqi_pm25) if aqi_pm25 is not None else None,
            "hora_medicion": data.get("time", {}).get("s"),
            "estacion_waqi": data.get("city", {}).get("name"),
        }
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("Fallo la llamada a WAQI: %s", exc)
        return None
