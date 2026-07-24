"""Panel 3: pronostico de PM2.5 por estacion (serie diaria).

Modelo: suavizado exponencial Holt-Winters (statsmodels), una serie por
estacion (regla de negocio 4). Se compara contra una media movil de 7 dias
como baseline, con MAPE y RMSE sobre un holdout final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def station_series(
    df_daily: pd.DataFrame, station: str, max_days: int = 1095
) -> pd.Series:
    """Serie diaria continua de PM2.5 de una estacion.

    Se interpolan todos los huecos para conservar la frecuencia diaria (que
    Holt-Winters necesita para la estacionalidad semanal) y se limita a los
    ultimos `max_days` (~3 anios) para modelar el regimen reciente y no
    arrastrar cambios de sensor/estacion de anios antiguos.
    """
    serie = (
        df_daily[df_daily["estacion"] == station]
        .set_index("fecha_dia")["pm25"]
        .asfreq("D")
        .interpolate(limit_direction="both")
    )
    return serie.iloc[-max_days:]


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


def evaluate_and_forecast_arima(
    serie: pd.Series, horizon: int = 7, test_days: int = 30
) -> dict:
    """Evalua ARIMA en holdout y pronostica horizon dias.

    Intenta (1,1,1) y degrada a ordenes mas simples si el ajuste falla.
    """
    if len(serie) < test_days + 60:
        raise ValueError(
            f"Serie demasiado corta ({len(serie)} dias) para evaluar con {test_days} dias de test."
        )
    train, test = serie.iloc[:-test_days], serie.iloc[-test_days:]

    def _fit(s: pd.Series) -> object:
        for order in [(1, 1, 1), (0, 1, 1), (1, 1, 0), (0, 1, 0)]:
            try:
                return _ARIMA(s, order=order).fit()
            except Exception:
                continue
        raise ValueError("ARIMA no convergio con ningun orden.")

    fit = _fit(train)
    pred_test = pd.Series(fit.forecast(steps=test_days).values, index=test.index).clip(lower=0)

    final_fit = _fit(serie)
    future_idx = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.Series(final_fit.forecast(steps=horizon).values, index=future_idx).clip(lower=0)

    return {
        "train": train,
        "test": test,
        "pred_test": pred_test,
        "forecast": future,
        "mape": _mape(test.values, pred_test.values),
        "rmse": _rmse(test.values, pred_test.values),
    }
