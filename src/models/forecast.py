"""Panel 3: pronostico de PM2.5 por estacion (serie diaria).

Modelo: suavizado exponencial Holt-Winters (statsmodels), una serie por
estacion (regla de negocio 4). Se compara contra una media movil de 7 dias
como baseline, con MAPE y RMSE sobre un holdout final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def station_series(df_daily: pd.DataFrame, station: str) -> pd.Series:
    """Serie diaria continua de PM2.5 de una estacion (huecos interpolados)."""
    serie = (
        df_daily[df_daily["estacion"] == station]
        .set_index("fecha_dia")["pm25"]
        .asfreq("D")
        .interpolate(limit=7)
        .dropna()
    )
    return serie


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_holt_winters(train: pd.Series) -> ExponentialSmoothing:
    """Holt-Winters con estacionalidad semanal; degrada a modelo simple si falla."""
    try:
        return ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=7
        ).fit()
    except (ValueError, np.linalg.LinAlgError):
        return ExponentialSmoothing(train, trend="add").fit()


def evaluate_and_forecast(
    serie: pd.Series, horizon: int = 7, test_days: int = 30
) -> dict:
    """Evalua en holdout (ultimos `test_days`) y pronostica `horizon` dias.

    Devuelve: metricas del modelo y del baseline, prediccion sobre el test
    (para graficar) y el pronostico futuro con la serie completa reajustada.
    """
    if len(serie) < test_days + 60:
        raise ValueError(
            f"Serie demasiado corta ({len(serie)} dias) para evaluar con {test_days} dias de test."
        )
    train, test = serie.iloc[:-test_days], serie.iloc[-test_days:]

    model = _fit_holt_winters(train)
    pred_test = model.forecast(test_days)

    # Baseline: media movil de 7 dias, actualizada con valores reales (rolling origin)
    history = train.copy()
    baseline_vals = []
    for real in test:
        baseline_vals.append(history.iloc[-7:].mean())
        history = pd.concat([history, pd.Series([real])], ignore_index=True)
    baseline = pd.Series(baseline_vals, index=test.index)

    final_model = _fit_holt_winters(serie)
    future = final_model.forecast(horizon).clip(lower=0)

    return {
        "train": train,
        "test": test,
        "pred_test": pred_test,
        "baseline_test": baseline,
        "forecast": future,
        "mape_modelo": _mape(test.values, pred_test.values),
        "rmse_modelo": _rmse(test.values, pred_test.values),
        "mape_baseline": _mape(test.values, baseline.values),
        "rmse_baseline": _rmse(test.values, baseline.values),
    }
