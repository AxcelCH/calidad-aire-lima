"""Comparacion de modelos por consola (sin UI) — genera el reporte de pruebas.

Uso: python scripts/run_experiments.py [ruta_csv_opcional]
Entrena clustering, clasificadores y pronostico sobre el CSV real de SENAMHI
e imprime todas las metricas. Sirve para el reporte y para probar cambios
sin levantar Streamlit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.clean import aggregate_daily, clean_hourly
from src.models.classifier import best_model_name, build_features, train_compare
from src.models.clustering import cluster_profiles, elbow_and_silhouette, run_kmeans
from src.models.forecast import evaluate_and_forecast, station_series


def main() -> None:
    t0 = time.time()
    if len(sys.argv) > 1:
        raw = pd.read_csv(sys.argv[1], sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    else:
        from src.ingest.senamhi import load_raw

        raw = load_raw()

    print("=" * 70)
    print("1) LIMPIEZA")
    hourly, report = clean_hourly(raw)
    for key, val in report.items():
        print(f"   {key}: {val}")
    daily = aggregate_daily(hourly)
    print(f"   dias-estacion tras agregacion diaria (>=12h de PM2.5): {len(daily):,}")

    print("=" * 70)
    print("2) CLUSTERING (K-means)")
    elbow = elbow_and_silhouette(daily)
    print(elbow.to_string(index=False))
    best_k = int(elbow.loc[elbow["silueta"].idxmax(), "k"])
    print(f"   k sugerido por silueta: {best_k}")
    clusters = run_kmeans(daily, best_k)
    print(cluster_profiles(clusters).to_string(index=False))

    print("=" * 70)
    print("3) CLASIFICACION (RF vs XGBoost) — objetivo: excede_pm25")
    features = build_features(daily)
    balance = features["excede_pm25"].mean()
    print(f"   filas: {len(features):,} · % clase 'excede': {balance:.1%}")
    results = train_compare(features)
    print(f"   SMOTE aplicado: {results['smote_aplicado']}")
    for name, res in results["modelos"].items():
        print(
            f"   {name:14s} acc={res['accuracy']:.4f} prec={res['precision']:.4f} "
            f"rec={res['recall']:.4f} f1={res['f1']:.4f} auc={res['roc_auc']:.4f}"
        )
        print(f"     matriz de confusion:\n{res['matriz_confusion']}")
    print(f"   MEJOR MODELO: {best_model_name(results)}")

    print("=" * 70)
    print("4) PRONOSTICO (Holt-Winters vs media movil 7d) — PM2.5 diario")
    for station in sorted(daily["estacion"].unique()):
        serie = station_series(daily, station)
        try:
            res = evaluate_and_forecast(serie)
            gana = "modelo" if res["mape_modelo"] < res["mape_baseline"] else "baseline"
            print(
                f"   {station:28s} n={len(serie):5d} | modelo MAPE={res['mape_modelo']:6.1f}% "
                f"RMSE={res['rmse_modelo']:5.1f} | baseline MAPE={res['mape_baseline']:6.1f}% "
                f"RMSE={res['rmse_baseline']:5.1f} | gana: {gana}"
            )
        except ValueError as exc:
            print(f"   {station:28s} omitida ({exc})")

    print("=" * 70)
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
