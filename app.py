"""Dashboard de calidad del aire en Lima Metropolitana — Mineria de Datos 2026-I.

Punto de entrada unico: arma los 4 paneles con st.tabs().
Datos reales: CSV historico de SENAMHI (datosabiertos.gob.pe) + API WAQI en vivo.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import roc_curve

from src.data.clean import aggregate_daily, clean_hourly
from src.data.eca import ECA_PM25_24H
from src.db.supabase_client import ConsultasRepo
from src.ingest.senamhi import load_raw
from src.ingest.waqi import STATIONS_WAQI, get_live_reading
from src.models.classifier import best_model_name, best_tree_name, build_features, train_compare
from src.models.clustering import cluster_profiles, elbow_and_silhouette, run_dbscan, run_kmeans
from src.models.forecast import evaluate_and_forecast, evaluate_and_forecast_arima, station_series

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Calidad del Aire — Lima", layout="wide")

# Paleta validada (colorblind-safe, orden categorico fijo) compartida por todos los graficos.
CATEGORICAL_COLORS = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
DIVERGING_SCALE = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
COLOR_MUTED = "#898781"
COLOR_SECONDARY = "#52514e"
COLOR_CRITICAL = "#d03b3b"


# ----------------------------- Carga cacheada -----------------------------
@st.cache_data(show_spinner="Descargando/leyendo CSV historico de SENAMHI...")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """CSV crudo -> limpio horario -> promedio diario. Cacheado en memoria."""
    raw = load_raw()
    hourly, report = clean_hourly(raw)
    daily = aggregate_daily(hourly)
    return hourly, daily, report


@st.cache_data(show_spinner="Calculando codo y silueta...")
def cached_elbow(daily: pd.DataFrame) -> pd.DataFrame:
    return elbow_and_silhouette(daily)


@st.cache_data(show_spinner="Entrenando K-means...")
def cached_kmeans(daily: pd.DataFrame, k: int) -> pd.DataFrame:
    return run_kmeans(daily, k)


@st.cache_data(show_spinner="Entrenando DBSCAN...")
def cached_dbscan(daily: pd.DataFrame, eps: float, min_samples: int) -> tuple[pd.DataFrame, dict]:
    return run_dbscan(daily, eps, min_samples)


@st.cache_data(show_spinner="Construyendo features supervisadas...")
def cached_features(daily: pd.DataFrame) -> pd.DataFrame:
    return build_features(daily)


@st.cache_data(show_spinner="Ajustando ARIMA (1,1,1)...")
def cached_arima(daily: pd.DataFrame, station: str, horizon: int) -> dict | None:
    serie = station_series(daily, station)
    try:
        return evaluate_and_forecast_arima(serie, horizon=horizon)
    except Exception:
        return None


@st.cache_resource(show_spinner="Entrenando MLP, Random Forest y XGBoost...")
def cached_training(
    _features: pd.DataFrame,
    test_size: float,
    n_estimators: int,
    learning_rate: float,
    mlp_epochs: int,
    mlp_activation: str,
    cache_key: str,
) -> dict:
    return train_compare(_features, test_size, n_estimators, learning_rate, mlp_epochs, mlp_activation)


@st.cache_resource
def get_repo() -> ConsultasRepo:
    return ConsultasRepo()


# ----------------------------- Paneles -----------------------------
def panel_eda(hourly: pd.DataFrame, daily: pd.DataFrame, report: dict) -> None:
    st.header("Panel 1 — Analisis exploratorio y clustering")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        (c1, c2, c3, c4),
        ("Registros horarios validos", "% descartado (nulos)", "Estaciones", "Cobertura"),
        (
            f"{report['registros_validos']:,}",
            f"{report['pct_descartado']}%",
            len(report["estaciones"]),
            f"{report['fecha_min']} → {report['fecha_max']}",
        ),
    ):
        with col.container(border=True):
            st.metric(label, value)

    st.subheader("Estadisticas descriptivas (promedios diarios)")
    st.dataframe(
        daily.groupby("estacion")[["pm10", "pm25", "no2"]]
        .agg(["mean", "median", "std"])
        .round(1),
        use_container_width=True,
    )

    st.subheader("Mapa de estaciones de monitoreo — Lima Metropolitana")
    coord_cols = [c for c in ["latitud", "longitud"] if c in hourly.columns]
    if len(coord_cols) == 2:
        coords = (
            hourly.dropna(subset=["latitud", "longitud"])
            .assign(
                latitud=lambda d: pd.to_numeric(d["latitud"], errors="coerce"),
                longitud=lambda d: pd.to_numeric(d["longitud"], errors="coerce"),
            )
            .dropna(subset=["latitud", "longitud"])
            .groupby("estacion")[["latitud", "longitud"]]
            .first()
            .reset_index()
        )
        avg_pm25 = (
            daily.groupby("estacion")["pm25"].mean().round(1)
            .reset_index().rename(columns={"pm25": "PM2.5 promedio (µg/m³)"})
        )
        map_df = coords.merge(avg_pm25, on="estacion").dropna()
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df, lat="latitud", lon="longitud",
                color="PM2.5 promedio (µg/m³)", size="PM2.5 promedio (µg/m³)",
                hover_name="estacion",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 80],
                size_max=25, zoom=10,
                mapbox_style="open-street-map",
                title="PM2.5 promedio historico por estacion (ECA = 50 µg/m³)",
            )
            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Coordenadas geograficas no disponibles en este dataset.")

    st.subheader("Distribuciones y outliers (regla 1.5·IQR)")
    pol = st.selectbox("Contaminante", ["pm25", "pm10", "no2"], key="eda_pol")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.histogram(
                daily, x=pol, nbins=60, title=f"Histograma de {pol.upper()} (diario)",
                color_discrete_sequence=[CATEGORICAL_COLORS[0]],
            ),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.box(
                daily, x="estacion", y=pol, title=f"Boxplot de {pol.upper()} por estacion",
                color_discrete_sequence=[CATEGORICAL_COLORS[0]],
            ),
            use_container_width=True,
        )

    st.subheader("Outliers detectados (regla 1.5·IQR)")
    outlier_rows = []
    for cont in ["pm25", "pm10", "no2"]:
        if cont not in daily.columns:
            continue
        serie = daily[cont].dropna()
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())
        outlier_rows.append({
            "Contaminante": cont.upper(),
            "Q1": round(q1, 1),
            "Q3": round(q3, 1),
            "IQR": round(iqr, 1),
            "Limite inferior": round(q1 - 1.5 * iqr, 1),
            "Limite superior": round(q3 + 1.5 * iqr, 1),
            "Outliers": n_out,
            "% del total": f"{100 * n_out / len(serie):.1f}%",
        })
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)

    st.subheader("Correlacion entre contaminantes")
    corr = daily[["pm10", "pm25", "no2"]].corr().round(2)
    st.plotly_chart(
        px.imshow(corr, text_auto=True, color_continuous_scale=DIVERGING_SCALE, zmin=-1, zmax=1),
        use_container_width=True,
    )

    st.subheader("Clustering K-means")
    elbow = cached_elbow(daily)
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(
            px.line(
                elbow, x="k", y="inercia", markers=True, title="Metodo del codo",
                color_discrete_sequence=[CATEGORICAL_COLORS[0]],
            ),
            use_container_width=True,
        )
    with col_d:
        st.plotly_chart(
            px.line(
                elbow, x="k", y="silueta", markers=True, title="Coeficiente de silueta",
                color_discrete_sequence=[CATEGORICAL_COLORS[1]],
            ),
            use_container_width=True,
        )

    k = st.slider("Numero de clusters (k)", 2, 8, 3, key="k_clusters")  # hiperparametro en vivo
    clusters = cached_kmeans(daily, k)
    st.plotly_chart(
        px.scatter(
            clusters, x="pm25", y="pm10", color="cluster", hover_data=["estacion", "no2"],
            title=f"Clusters de dias segun contaminacion (k={k})",
            color_discrete_sequence=CATEGORICAL_COLORS,
        ),
        use_container_width=True,
    )
    st.caption("Perfil promedio de cada cluster:")
    st.dataframe(cluster_profiles(clusters), use_container_width=True)

    st.subheader("DBSCAN — clusters de forma arbitraria y deteccion de outliers")
    st.markdown(
        "DBSCAN no exige fijar k: agrupa por densidad y marca como **outlier** (ruido) "
        "los dias que no pertenecen a ninguna region densa. Se compara su silueta con la de K-means."
    )
    col_e, col_f = st.columns(2)
    eps = col_e.slider("eps (radio de vecindad, datos escalados)", 0.2, 2.0, 0.7, 0.1)  # en vivo
    min_samples = col_f.slider("min_samples", 5, 50, 15, 5)                              # en vivo
    db_clusters, db_summary = cached_dbscan(daily, eps, min_samples)

    sil_kmeans = float(elbow.loc[elbow["k"] == k, "silueta"].iloc[0])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters DBSCAN", db_summary["n_clusters"])
    m2.metric("Outliers (ruido)", f"{db_summary['n_outliers']:,} ({db_summary['pct_outliers']}%)")
    sil_db = db_summary["silueta"]
    m3.metric("Silueta DBSCAN", "—" if pd.isna(sil_db) else f"{sil_db:.3f}")
    m4.metric(f"Silueta K-means (k={k})", f"{sil_kmeans:.3f}")

    st.plotly_chart(
        px.scatter(
            db_clusters, x="pm25", y="pm10", color="cluster", hover_data=["estacion", "no2"],
            title=f"DBSCAN (eps={eps}, min_samples={min_samples}) — 'outlier' = dias atipicos",
            category_orders={"cluster": sorted(db_clusters["cluster"].unique())},
        ),
        use_container_width=True,
    )
    if pd.isna(sil_db):
        st.caption("Con estos parametros DBSCAN encontro menos de 2 clusters: sube eps o baja min_samples.")
    else:
        mejor_alg = "K-means" if sil_kmeans >= sil_db else "DBSCAN"
        st.caption(
            f"Comparacion de silueta (sin contar el ruido de DBSCAN): **{mejor_alg}** separa mejor. "
            "DBSCAN aporta ademas la deteccion de dias-outlier que K-means fuerza dentro de un cluster."
        )

    st.subheader("Excedencias del ECA-PM2.5 por mes y estacion")
    exc = daily.copy()
    exc["mes"] = exc["fecha_dia"].dt.to_period("M").dt.to_timestamp()
    exc["excede"] = (exc["pm25"] > ECA_PM25_24H).astype(int)
    monthly_exc = (
        exc.groupby(["estacion", "mes"])["excede"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"excede": "pct_excede"})
    )
    fig_exc = px.line(
        monthly_exc, x="mes", y="pct_excede", color="estacion", markers=True,
        title="% de días por mes que superan el ECA de PM2.5 (50 µg/m³)",
        labels={"mes": "Mes", "pct_excede": "% días con excedencia", "estacion": "Estación"},
    )
    fig_exc.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% días")
    st.plotly_chart(fig_exc, use_container_width=True)

    st.subheader("Patron de contaminacion: hora del dia vs dia de la semana")
    if "fecha_hora" in hourly.columns and "pm25" in hourly.columns:
        heat_df = hourly.dropna(subset=["pm25"]).copy()
        heat_df["hora"] = heat_df["fecha_hora"].dt.hour
        heat_df["dia"] = heat_df["fecha_hora"].dt.dayofweek
        pivot_heat = (
            heat_df.groupby(["dia", "hora"])["pm25"]
            .mean()
            .round(1)
            .unstack(level="hora")
        )
        _day_names = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
        pivot_heat.index = [_day_names.get(d, str(d)) for d in pivot_heat.index]
        fig_heat = px.imshow(
            pivot_heat,
            title="PM2.5 promedio (µg/m³) por hora y dia de la semana — todas las estaciones",
            labels={"x": "Hora del dia", "y": "Dia", "color": "PM2.5 (µg/m³)"},
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            text_auto=".0f",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Los patrones horarios reflejan los picos de trafico de Lima (7-9 am y 6-8 pm).")


def panel_predictivo(daily: pd.DataFrame) -> None:
    st.header("Panel 2 — Modelo predictivo: excedencia del ECA de PM2.5")
    st.markdown(
        f"**Variable objetivo:** `excede_pm25` = 1 si el promedio diario de PM2.5 "
        f"supera el ECA peruano de **{ECA_PM25_24H} µg/m³** (D.S. 003-2017-MINAM). "
        "Las features usan solo informacion de dias anteriores (sin fuga de datos)."
    )

    col1, col2, col3 = st.columns(3)
    test_size = col1.slider("test_size", 0.1, 0.4, 0.2, 0.05)          # en vivo
    n_estimators = col2.slider("n_estimators", 50, 500, 200, 50)       # en vivo
    learning_rate = col3.slider("learning_rate (XGBoost)", 0.01, 0.5, 0.1, 0.01)  # en vivo
    col4, col5 = st.columns(2)
    mlp_epochs = col4.slider("epochs / max_iter (MLP)", 10, 500, 200, 10)              # en vivo
    mlp_activation = col5.selectbox("activation (MLP)", ["relu", "logistic", "tanh"])  # en vivo

    features = cached_features(daily)
    balance = features["excede_pm25"].value_counts(normalize=True).round(3)
    st.caption(
        f"Dataset supervisado: {len(features):,} filas · balance de clases: "
        f"{balance.get(1, 0):.1%} excede / {balance.get(0, 0):.1%} no excede"
    )

    key = f"{test_size}-{n_estimators}-{learning_rate}-{mlp_epochs}-{mlp_activation}"
    results = cached_training(features, test_size, n_estimators, learning_rate, mlp_epochs, mlp_activation, key)
    if results["smote_aplicado"]:
        st.info(
            f"Clase minoritaria = {results['ratio_minoritaria_train']:.1%} (< 20%): "
            "se aplico **SMOTE** solo sobre el conjunto de entrenamiento."
        )

    metric_rows = [
        {"Modelo": name, **{m: round(res[m], 4) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}
        for name, res in results["modelos"].items()
    ]
    st.dataframe(pd.DataFrame(metric_rows).set_index("Modelo"), use_container_width=True)

    cols = st.columns(len(results["modelos"]))
    for col, (name, res) in zip(cols, results["modelos"].items()):
        with col:
            st.plotly_chart(
                px.imshow(
                    res["matriz_confusion"], text_auto=True, color_continuous_scale=SEQUENTIAL_BLUE,
                    x=["Pred: no excede", "Pred: excede"], y=["Real: no excede", "Real: excede"],
                    title=f"Matriz de confusion — {name}",
                ),
                use_container_width=True,
            )

    with st.expander("Metricas por clase (precision, recall, F1 de 'no excede' y 'excede')"):
        for name, res in results["modelos"].items():
            rep = pd.DataFrame(res["reporte_clases"]).T.loc[
                ["no excede", "excede"], ["precision", "recall", "f1-score", "support"]
            ].round(3)
            st.markdown(f"**{name}** — accuracy global: {res['accuracy']:.3f}")
            st.dataframe(rep, use_container_width=True)
        st.caption(
            "Para alertas de salud publica el error mas costoso es el falso negativo "
            "(no avisar un dia que SI excede el ECA): por eso se vigila el recall de la clase 'excede'."
        )

    st.subheader("Curvas ROC — comparacion de modelos")
    fig_roc = go.Figure()
    for name, res in results["modelos"].items():
        fpr, tpr, _ = roc_curve(results["y_test"], res["y_proba"])
        fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['roc_auc']:.3f})")
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Clasificador aleatorio",
                        line={"dash": "dot", "color": "gray"})
    fig_roc.update_layout(
        xaxis_title="Tasa de falsos positivos (FPR)",
        yaxis_title="Tasa de verdaderos positivos (TPR)",
        legend={"yanchor": "bottom", "y": 0.05, "xanchor": "right", "x": 0.95},
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    best = best_model_name(results)
    others = ", ".join(
        f"{n} (F1={results['modelos'][n]['f1']:.3f})" for n in results["modelos"] if n != best
    )
    st.success(
        f"**Mejor modelo: {best}** — F1 = {results['modelos'][best]['f1']:.3f} y "
        f"ROC-AUC = {results['modelos'][best]['roc_auc']:.3f}, frente a {others}. "
        "En este problema el costo de un falso negativo (no alertar un dia que si excede el ECA) "
        "es mayor que el de una falsa alarma, por eso se prioriza F1/recall de la clase 'excede'."
    )

    # SHAP TreeExplainer y feature_importances_ solo aplican a modelos de arboles
    best_tree = best_tree_name(results)
    if best != best_tree:
        st.caption(
            f"La interpretabilidad de abajo usa **{best_tree}** (mejor modelo de arboles): "
            "el MLP es una caja negra sin feature_importances_ ni TreeExplainer."
        )
    model = results["modelos"][best_tree]["modelo"]
    X_test = results["X_test"]

    st.subheader(f"Importancia de variables — {best_tree}")
    importances = (
        pd.Series(model.feature_importances_, index=X_test.columns)
        .sort_values(ascending=True)
        .tail(15)
    )
    fig_fi = px.bar(
        importances, orientation="h",
        title=f"Top 15 variables mas importantes — {best_tree}",
        labels={"value": "Importancia", "index": "Variable"},
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption(
        "La importancia de arboles mide reduccion de impureza en nodos. "
        "El SHAP (abajo) da el impacto real sobre cada prediccion individual."
    )

    st.subheader("Interpretabilidad con SHAP")
    with st.spinner("Calculando valores SHAP..."):
        import shap

        sample = X_test.iloc[:300]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):  # RandomForest binario devuelve lista
            shap_values = shap_values[1]
        if getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 1]

        fig_summary = plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(fig_summary, clear_figure=True)
        st.caption(
            "Lectura global: los rezagos recientes de PM2.5 (dia anterior y media movil de 7 dias) "
            "son los que mas empujan la prediccion hacia 'excede'; valores altos (rojo) aumentan el riesgo."
        )

        st.markdown("**Explicacion de un caso concreto (force plot):**")
        idx = st.number_input("Fila del conjunto de test a explicar", 0, len(sample) - 1, 0)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, tuple)) or getattr(base_value, "ndim", 0) == 1:
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        shap.plots.force(base_value, shap_values[int(idx)], sample.iloc[int(idx)], matplotlib=True, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)


def panel_pronostico(daily: pd.DataFrame) -> None:
    st.header("Panel 3 — Pronostico de PM2.5 (serie diaria por estacion)")

    station = st.selectbox("Estacion", sorted(daily["estacion"].unique()), key="fc_station")
    horizon = st.slider("Dias a pronosticar", 4, 14, 7)

    serie = station_series(daily, station)
    try:
        res_hw = evaluate_and_forecast(serie, horizon=horizon)
    except ValueError as exc:
        st.warning(f"No se puede pronosticar esta estacion: {exc}")
        return

    res_arima = cached_arima(daily, station, horizon)

    # --- Tarjetas de metricas ---
    if res_arima is not None:
        labels = [
            "MAPE Holt-Winters", "RMSE Holt-Winters", "MAPE ARIMA(1,1,1)", "RMSE ARIMA(1,1,1)",
            "MAPE baseline MM-7d", "RMSE baseline",
        ]
        values = [
            f"{res_hw['mape_modelo']:.1f}%", f"{res_hw['rmse_modelo']:.1f}",
            f"{res_arima['mape']:.1f}%", f"{res_arima['rmse']:.1f}",
            f"{res_hw['mape_baseline']:.1f}%", f"{res_hw['rmse_baseline']:.1f}",
        ]
        cols = st.columns(6)
    else:
        labels = ["MAPE Holt-Winters", "RMSE Holt-Winters", "MAPE baseline MM-7d", "RMSE baseline"]
        values = [
            f"{res_hw['mape_modelo']:.1f}%", f"{res_hw['rmse_modelo']:.1f}",
            f"{res_hw['mape_baseline']:.1f}%", f"{res_hw['rmse_baseline']:.1f}",
        ]
        cols = st.columns(4)
    for col, label, value in zip(cols, labels, values):
        with col.container(border=True):
            st.metric(label, value)

    # --- Grafico unificado ---
    hist = pd.concat([res_hw["train"].iloc[-120:], res_hw["test"]])
    trend = hist.rolling(30, min_periods=10).mean()
    # Ancla cada pronostico al ultimo dato conocido para que la linea no quede flotando.
    forecast_hw_plot = pd.concat([res_hw["test"].iloc[[-1]], res_hw["forecast"]])

    fig = go.Figure()
    fig.add_scatter(
        x=hist.index, y=hist.values, name="Historico", mode="lines",
        line={"color": COLOR_MUTED, "width": 2},
    )
    fig.add_scatter(
        x=trend.index, y=trend.values, name="Tendencia (media movil 30d)", mode="lines",
        line={"color": COLOR_SECONDARY, "width": 2},
    )
    fig.add_scatter(
        x=res_hw["test"].index, y=res_hw["pred_test"].values, name="Holt-Winters (holdout)", mode="lines",
        line={"color": CATEGORICAL_COLORS[0], "width": 2},
    )
    fig.add_scatter(
        x=res_hw["test"].index, y=res_hw["baseline_test"].values, name="Baseline MM-7d", mode="lines",
        line={"color": CATEGORICAL_COLORS[1], "width": 2, "dash": "dot"},
    )
    if res_arima is not None:
        forecast_arima_plot = pd.concat([res_arima["test"].iloc[[-1]], res_arima["forecast"]])
        fig.add_scatter(
            x=res_arima["test"].index, y=res_arima["pred_test"].values, name="ARIMA(1,1,1) (holdout)",
            mode="lines", line={"color": CATEGORICAL_COLORS[4], "width": 2},
        )
    fig.add_scatter(
        x=forecast_hw_plot.index, y=forecast_hw_plot.values, name=f"Pronostico HW +{horizon}d",
        mode="lines+markers", line={"color": CATEGORICAL_COLORS[2], "width": 2}, marker={"size": 8},
    )
    if res_arima is not None:
        fig.add_scatter(
            x=forecast_arima_plot.index, y=forecast_arima_plot.values, name=f"Pronostico ARIMA +{horizon}d",
            mode="lines+markers", line={"color": CATEGORICAL_COLORS[3], "width": 2, "dash": "dash"},
            marker={"size": 8},
        )
    fig.add_vline(x=res_hw["test"].index[-1], line_dash="dot", line_color=COLOR_MUTED)
    fig.add_annotation(
        x=res_hw["test"].index[-1], y=1, yref="paper", yanchor="bottom",
        text="Hoy", showarrow=False, font={"color": COLOR_MUTED, "size": 12},
    )
    fig.add_hline(
        y=ECA_PM25_24H, line_dash="dash", line_color=COLOR_CRITICAL,
        annotation_text=f"ECA {ECA_PM25_24H} µg/m³", annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"PM2.5 diario — {station} (Holt-Winters vs ARIMA vs baseline)",
        yaxis_title="µg/m³",
        xaxis_title=None,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0},
        margin={"t": 80},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Conclusion ---
    if res_arima is not None:
        mejor = "Holt-Winters" if res_hw["mape_modelo"] <= res_arima["mape"] else "ARIMA(1,1,1)"
        st.success(
            f"**Mejor modelo en holdout: {mejor}** — "
            f"MAPE HW={res_hw['mape_modelo']:.1f}% · "
            f"MAPE ARIMA={res_arima['mape']:.1f}% · "
            f"MAPE baseline={res_hw['mape_baseline']:.1f}%"
        )
    elif res_hw["mape_modelo"] < res_hw["mape_baseline"]:
        st.success("El modelo Holt-Winters supera a la media movil de 7 dias en el holdout.")
    else:
        st.warning("La media movil de 7 dias es competitiva: la serie es muy persistente en este periodo.")

    with st.expander("Ver pronostico en tabla"):
        tabla = pd.DataFrame({"fecha": res_hw["forecast"].index.date, "pronostico_hw_ugm3": res_hw["forecast"].values.round(1)})
        if res_arima is not None:
            tabla["pronostico_arima_ugm3"] = res_arima["forecast"].values.round(1)
        st.dataframe(tabla, use_container_width=True, hide_index=True)


def panel_crud(daily: pd.DataFrame, features: pd.DataFrame, results: dict) -> None:
    st.header("Panel 4 — Consultas: prediccion vs. valor real (API WAQI)")
    repo = get_repo()
    st.caption(
        f"Base de datos activa: **{repo.backend.upper()}**"
        + ("" if repo.backend == "supabase" else " (fallback local; configura SUPABASE_URL/ANON_KEY para usar Postgres)")
    )

    best = best_model_name(results)
    model = results["modelos"][best]["modelo"]
    x_cols = list(results["X_test"].columns)

    modo = st.radio(
        "Modo de consulta",
        ["Automatica (ultimo dia historico + WAQI en vivo)", "Manual (ingresa tus propios datos)"],
        horizontal=True,
    )

    if modo.startswith("Automatica"):
        with st.form("form_consulta"):
            st.markdown("**Nueva consulta**: predice si MAÑANA excedera el ECA de PM2.5 en la estacion elegida y contrasta con el dato en vivo de WAQI.")
            station = st.selectbox("Estacion", sorted(set(daily["estacion"].unique()) & set(STATIONS_WAQI)))
            submitted = st.form_submit_button("Predecir y guardar")

        if submitted:
            # Ultima fila de features disponible para la estacion elegida
            est_col = f"est_{station}"
            if est_col not in features.columns:
                st.error("No hay features historicas para esa estacion.")
                return
            rows = features[features[est_col] == 1].sort_values("fecha_dia")
            if rows.empty:
                st.error("No hay datos suficientes de esa estacion.")
                return
            x_last = rows.drop(columns=["excede_pm25", "fecha_dia"]).astype(float).iloc[[-1]]
            proba = float(model.predict_proba(x_last)[0, 1])

            live = get_live_reading(station)
            # WAQI puede responder OK sin traer PM2.5 (esa estacion solo reporta otros
            # contaminantes en ese momento): valor_real depende del dato, no de la respuesta.
            valor_real = live.get("pm25_estimado_ugm3") if live else None
            ok = repo.guardar(
                estacion=station,
                inputs={"tipo": "automatica", "fecha_features": str(rows["fecha_dia"].iloc[-1].date()), "modelo": best},
                valor_predicho=proba,
                valor_real=valor_real,
                fuente_en_vivo=valor_real is not None,
            )
            c1, c2 = st.columns(2)
            c1.metric("Prob. de exceder ECA (modelo)", f"{proba:.1%}")
            if valor_real is not None:
                c2.metric(
                    "PM2.5 en vivo (WAQI)",
                    f"{valor_real} µg/m³",
                    help=f"AQI pm25={live['aqi_pm25']} medido {live['hora_medicion']} en {live['estacion_waqi']}",
                )
            elif live:
                c2.warning("WAQI no reporta PM2.5 para esta estacion en este momento: se guardo solo la prediccion.")
            else:
                c2.warning("API WAQI no respondio: se guardo solo la prediccion (fuente_en_vivo = false).")
            st.success("Consulta guardada." if ok else "No se pudo guardar la consulta.")
    else:
        st.markdown(
            "**Consulta manual**: ingresa las mediciones de los ultimos dias y el modelo "
            "predice si el dia elegido excedera el ECA de PM2.5. Los campos vienen "
            "prellenados con el ultimo dato historico de la estacion — puedes editarlos todos."
        )
        station_m = st.selectbox("Estacion", sorted(daily["estacion"].unique()), key="manual_station")

        # Defaults desde la ultima fila historica de la estacion (si existe)
        est_col_m = f"est_{station_m}"
        defaults = {}
        if est_col_m in features.columns:
            rows_m = features[features[est_col_m] == 1].sort_values("fecha_dia")
            if not rows_m.empty:
                defaults = rows_m.iloc[-1].to_dict()

        def _default(col: str, fallback: float, hi: float = 500.0) -> float:
            val = defaults.get(col)
            if val is None or pd.isna(val):
                return fallback
            return min(max(float(val), 0.0), hi)  # dentro del rango del number_input

        with st.form("form_consulta_manual"):
            fecha_obj = st.date_input("Fecha a predecir (define dia de semana y mes)")
            st.markdown("**PM2.5 (µg/m³) de los dias previos**")
            c1, c2, c3 = st.columns(3)
            pm25_l1 = c1.number_input("PM2.5 ayer (lag 1)", 0.0, 500.0, _default("pm25_lag1", 40.0), 0.5)
            pm25_l2 = c2.number_input("PM2.5 hace 2 dias", 0.0, 500.0, _default("pm25_lag2", 40.0), 0.5)
            pm25_l3 = c3.number_input("PM2.5 hace 3 dias", 0.0, 500.0, _default("pm25_lag3", 40.0), 0.5)
            st.markdown("**PM10 (µg/m³) de los dias previos**")
            c4, c5, c6 = st.columns(3)
            pm10_l1 = c4.number_input("PM10 ayer (lag 1)", 0.0, 800.0, _default("pm10_lag1", 80.0, 800.0), 0.5)
            pm10_l2 = c5.number_input("PM10 hace 2 dias", 0.0, 800.0, _default("pm10_lag2", 80.0, 800.0), 0.5)
            pm10_l3 = c6.number_input("PM10 hace 3 dias", 0.0, 800.0, _default("pm10_lag3", 80.0, 800.0), 0.5)
            st.markdown("**NO₂ (µg/m³) de los dias previos**")
            c7, c8, c9 = st.columns(3)
            no2_l1 = c7.number_input("NO₂ ayer (lag 1)", 0.0, 400.0, _default("no2_lag1", 20.0, 400.0), 0.5)
            no2_l2 = c8.number_input("NO₂ hace 2 dias", 0.0, 400.0, _default("no2_lag2", 20.0, 400.0), 0.5)
            no2_l3 = c9.number_input("NO₂ hace 3 dias", 0.0, 400.0, _default("no2_lag3", 20.0, 400.0), 0.5)
            media7 = st.number_input(
                "PM2.5 promedio de los ultimos 7 dias (µg/m³)",
                0.0, 500.0, _default("pm25_media7d", 40.0), 0.5,
            )
            submitted_m = st.form_submit_button("Predecir con datos manuales y guardar")

        if submitted_m:
            valores = {
                "pm25_lag1": pm25_l1, "pm25_lag2": pm25_l2, "pm25_lag3": pm25_l3,
                "pm10_lag1": pm10_l1, "pm10_lag2": pm10_l2, "pm10_lag3": pm10_l3,
                "no2_lag1": no2_l1, "no2_lag2": no2_l2, "no2_lag3": no2_l3,
                "pm25_media7d": media7,
                "dia_semana": float(fecha_obj.weekday()),
                "mes": float(fecha_obj.month),
            }
            x_manual = pd.DataFrame(0.0, index=[0], columns=x_cols)
            for col, val in valores.items():
                if col in x_manual.columns:
                    x_manual.loc[0, col] = val
            if est_col_m in x_manual.columns:
                x_manual.loc[0, est_col_m] = 1.0

            proba = float(model.predict_proba(x_manual)[0, 1])
            veredicto = "EXCEDE" if proba >= 0.5 else "no excede"
            ok = repo.guardar(
                estacion=station_m,
                inputs={"tipo": "manual", "fecha_objetivo": str(fecha_obj), "modelo": best, **valores},
                valor_predicho=proba,
                valor_real=None,
                fuente_en_vivo=False,
            )
            c1, c2 = st.columns(2)
            c1.metric("Prob. de exceder ECA (modelo)", f"{proba:.1%}")
            c2.metric("Veredicto (umbral 0.5)", veredicto)
            st.success(
                f"Consulta manual guardada (modelo {best})." if ok else "No se pudo guardar la consulta."
            )

    st.subheader("Consultas guardadas")
    df = repo.listar()
    if df.empty:
        st.info("Aun no hay consultas guardadas.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Exportar CSV", df.to_csv(index=False).encode(), "consultas.csv", "text/csv")

    col_e, col_d = st.columns(2)
    with col_e:
        st.markdown("**Editar observacion** (unico campo editable)")
        edit_id = st.selectbox("ID a editar", df["id"].tolist(), key="edit_id")
        nueva_obs = st.text_input("Observacion")
        if st.button("Guardar observacion"):
            st.success("Observacion actualizada.") if repo.editar_observacion(int(edit_id), nueva_obs) else st.error("Fallo la edicion.")
            st.rerun()
    with col_d:
        st.markdown("**Eliminar** (borrado logico)")
        del_id = st.selectbox("ID a eliminar", df["id"].tolist(), key="del_id")
        if st.button("Eliminar consulta"):
            st.success("Consulta marcada como eliminada.") if repo.eliminar(int(del_id)) else st.error("Fallo la eliminacion.")
            st.rerun()


# ----------------------------- Main -----------------------------
def main() -> None:
    st.title("Calidad del aire en Lima Metropolitana")
    st.markdown(
        "Proyecto de Minería de Datos (UNMSM-FISI, 2026-I). Datos reales: "
        "[SENAMHI / datosabiertos.gob.pe](https://www.datosabiertos.gob.pe/dataset/monitoreo-de-los-contaminantes-del-aire-en-lima-metropolitana-servicio-nacional-de)"
        " (histórico) y [API WAQI](https://aqicn.org) (en vivo)."
    )

    try:
        hourly, daily, report = get_data()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "No se pudo cargar el dataset de SENAMHI. Verifica la conexion o coloca el CSV "
            f"en `data_cache/`. Detalle: {exc}"
        )
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["EDA + Clustering", "Predictivo", "Pronostico", "Consultas (CRUD)"]
    )
    with tab1:
        panel_eda(hourly, daily, report)
    with tab2:
        panel_predictivo(daily)
    with tab3:
        panel_pronostico(daily)
    with tab4:
        # Reusa el modelo ya entrenado con los valores por defecto de los sliders
        features = cached_features(daily)
        results = cached_training(features, 0.2, 200, 0.1, 200, "relu", "0.2-200-0.1-200-relu")
        panel_crud(daily, features, results)


if __name__ == "__main__":
    main()
