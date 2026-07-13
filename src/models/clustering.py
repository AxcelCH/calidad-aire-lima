"""Panel 1: clustering K-means de perfiles de contaminacion.

Combina todas las estaciones (regla de negocio 4) para encontrar perfiles
de dias de contaminacion distintos. Incluye metodo del codo y silueta.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["pm10", "pm25", "no2"]


def _prepare(df_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra filas completas y escala las variables. Devuelve (df, X_escalado)."""
    cols = [c for c in FEATURES if c in df_daily.columns]
    data = df_daily.dropna(subset=cols).copy()
    X = StandardScaler().fit_transform(data[cols])
    return data, X


def elbow_and_silhouette(df_daily: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """Inercia (codo) y coeficiente de silueta para k en [k_min, k_max]."""
    _, X = _prepare(df_daily)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({
            "k": k,
            "inercia": km.inertia_,
            "silueta": silhouette_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def run_kmeans(df_daily: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Ejecuta K-means y devuelve el DataFrame diario con columna 'cluster'."""
    data, X = _prepare(df_daily)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    data["cluster"] = km.labels_.astype(str)
    return data


def cluster_profiles(df_clusters: pd.DataFrame) -> pd.DataFrame:
    """Perfil promedio de cada cluster (para interpretarlos en la UI)."""
    cols = [c for c in FEATURES if c in df_clusters.columns]
    return df_clusters.groupby("cluster")[cols].mean().round(1).reset_index()
