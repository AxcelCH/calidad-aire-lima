"""Panel 2: clasificacion de excedencia ECA de PM2.5 (RF vs XGBoost).

La variable objetivo es 'excede_pm25' del dia t. Para evitar fuga de
informacion (data leakage), las features solo usan datos de dias ANTERIORES
(rezagos y promedios moviles) mas variables de calendario.
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.eca import ECA_PM25_24H

LAGS = [1, 2, 3]


def build_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Construye el dataset supervisado a partir del promedio diario."""
    df = df_daily.sort_values(["estacion", "fecha_dia"]).copy()
    df["excede_pm25"] = (df["pm25"] > ECA_PM25_24H).astype(int)

    frames = []
    for _, grp in df.groupby("estacion"):
        g = grp.copy()
        for lag in LAGS:
            g[f"pm25_lag{lag}"] = g["pm25"].shift(lag)
            g[f"pm10_lag{lag}"] = g["pm10"].shift(lag) if "pm10" in g else pd.NA
            g[f"no2_lag{lag}"] = g["no2"].shift(lag) if "no2" in g else pd.NA
        g["pm25_media7d"] = g["pm25"].shift(1).rolling(7, min_periods=3).mean()
        frames.append(g)
    feat = pd.concat(frames)

    feat["dia_semana"] = feat["fecha_dia"].dt.dayofweek
    feat["mes"] = feat["fecha_dia"].dt.month
    feat = pd.get_dummies(feat, columns=["estacion"], prefix="est")

    feature_cols = [c for c in feat.columns if c.startswith(("pm25_lag", "pm10_lag", "no2_lag", "pm25_media7d", "dia_semana", "mes", "est_"))]
    feat = feat.dropna(subset=[c for c in feature_cols if not c.startswith("est_")])
    return feat[feature_cols + ["excede_pm25", "fecha_dia"]].reset_index(drop=True)


def train_compare(
    features: pd.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    use_smote: bool | None = None,
    random_state: int = 42,
) -> dict:
    """Entrena RF y XGBoost con el mismo split y devuelve metricas comparables.

    use_smote=None aplica la regla de negocio 5: SMOTE automatico si la clase
    minoritaria es < 20% (solo sobre train, nunca sobre test).
    """
    X = features.drop(columns=["excede_pm25", "fecha_dia"]).astype(float)
    y = features["excede_pm25"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    minority_ratio = y_train.mean() if y_train.mean() < 0.5 else 1 - y_train.mean()
    if use_smote is None:
        use_smote = minority_ratio < 0.20
    if use_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results: dict = {
        "smote_aplicado": bool(use_smote),
        "ratio_minoritaria_train": round(float(minority_ratio), 4),
        "X_test": X_test,
        "y_test": y_test,
        "modelos": {},
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results["modelos"][name] = {
            "modelo": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "matriz_confusion": confusion_matrix(y_test, y_pred),
        }
    return results


def best_model_name(results: dict) -> str:
    """Elige el mejor modelo por F1 (empate -> ROC-AUC)."""
    return max(
        results["modelos"],
        key=lambda n: (results["modelos"][n]["f1"], results["modelos"][n]["roc_auc"]),
    )
