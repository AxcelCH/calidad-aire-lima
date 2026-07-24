"""Tests del pipeline supervisado (anti-leakage) y del clustering DBSCAN."""
import numpy as np
import pandas as pd
import pytest

from src.models.classifier import build_features
from src.models.clustering import run_dbscan


def _daily_sintetico(n_dias: int = 30, estaciones=("A", "B")) -> pd.DataFrame:
    """Serie diaria sintetica y determinista para dos estaciones."""
    filas = []
    for est in estaciones:
        base = 30.0 if est == "A" else 60.0
        for i in range(n_dias):
            filas.append({
                "estacion": est,
                "fecha_dia": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "pm25": base + i,          # creciente: facil de verificar en los lags
                "pm10": 2 * (base + i),
                "no2": 10.0 + i,
            })
    return pd.DataFrame(filas)


class TestBuildFeatures:
    def test_lag1_es_el_pm25_del_dia_anterior(self):
        daily = _daily_sintetico()
        feat = build_features(daily)
        # Reconstruyo la estacion de cada fila desde las dummies
        fila = feat[feat["est_A"] == 1].sort_values("fecha_dia").iloc[-1]
        dia_previo = fila["fecha_dia"] - pd.Timedelta(days=1)
        pm25_previo = daily[
            (daily["estacion"] == "A") & (daily["fecha_dia"] == dia_previo)
        ]["pm25"].iloc[0]
        assert fila["pm25_lag1"] == pytest.approx(pm25_previo)

    def test_no_incluye_el_pm25_del_mismo_dia_como_feature(self):
        """Anti-leakage: ninguna columna de features contiene el valor del dia t."""
        feat = build_features(_daily_sintetico())
        feature_cols = [c for c in feat.columns if c not in ("excede_pm25", "fecha_dia")]
        assert "pm25" not in feature_cols
        assert all(c.startswith(("pm25_lag", "pm10_lag", "no2_lag", "pm25_media7d",
                                 "dia_semana", "mes", "est_")) for c in feature_cols)

    def test_target_respeta_el_umbral_eca(self):
        daily = _daily_sintetico()
        feat = build_features(daily)
        # La estacion B arranca en 60 (> 50): todas sus filas deben exceder
        assert (feat.loc[feat["est_B"] == 1, "excede_pm25"] == 1).all()

    def test_los_lags_no_cruzan_estaciones(self):
        """El lag de la primera fila valida de B no debe venir de A."""
        daily = _daily_sintetico()
        feat = build_features(daily)
        primera_b = feat[feat["est_B"] == 1].sort_values("fecha_dia").iloc[0]
        # pm25 de B siempre >= 60; si el lag viniera de A seria ~30
        assert primera_b["pm25_lag1"] >= 60.0


class TestDBSCAN:
    def test_detecta_outliers_como_ruido(self):
        rng = np.random.default_rng(42)
        normal = pd.DataFrame({
            "estacion": "A",
            "pm25": rng.normal(40, 2, 200),
            "pm10": rng.normal(80, 2, 200),
            "no2": rng.normal(20, 2, 200),
        })
        extremos = pd.DataFrame({
            "estacion": "A",
            "pm25": [300.0, 350.0],
            "pm10": [600.0, 700.0],
            "no2": [200.0, 250.0],
        })
        df = pd.concat([normal, extremos], ignore_index=True)
        etiquetado, resumen = run_dbscan(df, eps=0.7, min_samples=10)
        assert resumen["n_outliers"] >= 2
        assert (etiquetado["cluster"].iloc[-2:] == "outlier").all()

    def test_silueta_nan_si_hay_un_solo_cluster(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "estacion": "A",
            "pm25": rng.normal(40, 1, 100),
            "pm10": rng.normal(80, 1, 100),
            "no2": rng.normal(20, 1, 100),
        })
        _, resumen = run_dbscan(df, eps=2.0, min_samples=5)
        if resumen["n_clusters"] < 2:
            assert np.isnan(resumen["silueta"])
