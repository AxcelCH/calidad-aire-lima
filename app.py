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

from src.data.clean import aggregate_daily, clean_hourly
from src.data.eca import ECA_PM25_24H, label_exceedance
from src.db.supabase_client import ConsultasRepo
from src.ingest.senamhi import load_raw
from src.ingest.waqi import STATIONS_WAQI, get_live_reading
from src.models.classifier import best_model_name, build_features, train_compare
from src.models.clustering import cluster_profiles, elbow_and_silhouette, run_kmeans
from src.models.forecast import evaluate_and_forecast, station_series

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Calidad del Aire — Lima", page_icon="🌫️", layout="wide")


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


@st.cache_data(show_spinner="Construyendo features supervisadas...")
def cached_features(daily: pd.DataFrame) -> pd.DataFrame:
    return build_features(daily)


@st.cache_resource(show_spinner="Entrenando Random Forest y XGBoost...")
def cached_training(
    _features: pd.DataFrame, test_size: float, n_estimators: int, learning_rate: float, cache_key: str
) -> dict:
    return train_compare(_features, test_size, n_estimators, learning_rate)


@st.cache_resource
def get_repo() -> ConsultasRepo:
    return ConsultasRepo()


# ----------------------------- Paneles -----------------------------
def panel_eda(hourly: pd.DataFrame, daily: pd.DataFrame, report: dict) -> None:
    st.header("Panel 1 — Analisis exploratorio y clustering")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros horarios validos", f"{report['registros_validos']:,}")
    c2.metric("% descartado (nulos)", f"{report['pct_descartado']}%")
    c3.metric("Estaciones", len(report["estaciones"]))
    c4.metric("Cobertura", f"{report['fecha_min']} → {report['fecha_max']}")

    st.subheader("Estadisticas descriptivas (promedios diarios)")
    st.dataframe(
        daily.groupby("estacion")[["pm10", "pm25", "no2"]]
        .agg(["mean", "median", "std"])
        .round(1),
        use_container_width=True,
    )

    st.subheader("Distribuciones y outliers (regla 1.5·IQR)")
    pol = st.selectbox("Contaminante", ["pm25", "pm10", "no2"], key="eda_pol")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.histogram(daily, x=pol, nbins=60, title=f"Histograma de {pol.upper()} (diario)"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.box(daily, x="estacion", y=pol, title=f"Boxplot de {pol.upper()} por estacion"),
            use_container_width=True,
        )

    st.subheader("Correlacion entre contaminantes")
    corr = daily[["pm10", "pm25", "no2"]].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

    st.subheader("Clustering K-means")
    elbow = cached_elbow(daily)
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(px.line(elbow, x="k", y="inercia", markers=True, title="Metodo del codo"), use_container_width=True)
    with col_d:
        st.plotly_chart(px.line(elbow, x="k", y="silueta", markers=True, title="Coeficiente de silueta"), use_container_width=True)

    k = st.slider("Numero de clusters (k)", 2, 8, 3, key="k_clusters")  # hiperparametro en vivo
    clusters = cached_kmeans(daily, k)
    st.plotly_chart(
        px.scatter(
            clusters, x="pm25", y="pm10", color="cluster", hover_data=["estacion", "no2"],
            title=f"Clusters de dias segun contaminacion (k={k})",
        ),
        use_container_width=True,
    )
    st.caption("Perfil promedio de cada cluster:")
    st.dataframe(cluster_profiles(clusters), use_container_width=True)


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

    features = cached_features(daily)
    balance = features["excede_pm25"].value_counts(normalize=True).round(3)
    st.caption(
        f"Dataset supervisado: {len(features):,} filas · balance de clases: "
        f"{balance.get(1, 0):.1%} excede / {balance.get(0, 0):.1%} no excede"
    )

    key = f"{test_size}-{n_estimators}-{learning_rate}"
    results = cached_training(features, test_size, n_estimators, learning_rate, key)
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

    cols = st.columns(2)
    for col, (name, res) in zip(cols, results["modelos"].items()):
        with col:
            st.plotly_chart(
                px.imshow(
                    res["matriz_confusion"], text_auto=True, color_continuous_scale="Blues",
                    x=["Pred: no excede", "Pred: excede"], y=["Real: no excede", "Real: excede"],
                    title=f"Matriz de confusion — {name}",
                ),
                use_container_width=True,
            )

    best = best_model_name(results)
    other = next(n for n in results["modelos"] if n != best)
    st.success(
        f"**Mejor modelo: {best}** — F1 = {results['modelos'][best]['f1']:.3f} y "
        f"ROC-AUC = {results['modelos'][best]['roc_auc']:.3f}, frente a "
        f"F1 = {results['modelos'][other]['f1']:.3f} de {other}. "
        "En este problema el costo de un falso negativo (no alertar un dia que si excede el ECA) "
        "es mayor que el de una falsa alarma, por eso se prioriza F1/recall de la clase 'excede'."
    )

    st.subheader("Interpretabilidad con SHAP")
    with st.spinner("Calculando valores SHAP..."):
        import shap

        model = results["modelos"][best]["modelo"]
        X_test = results["X_test"]
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
        fig_force = plt.figure()
        shap.plots.force(base_value, shap_values[int(idx)], sample.iloc[int(idx)], matplotlib=True, show=False)
        st.pyplot(fig_force, clear_figure=True)


def panel_pronostico(daily: pd.DataFrame) -> None:
    st.header("Panel 3 — Pronostico de PM2.5 (serie diaria por estacion)")

    station = st.selectbox("Estacion", sorted(daily["estacion"].unique()), key="fc_station")
    horizon = st.slider("Dias a pronosticar", 4, 14, 7)

    serie = station_series(daily, station)
    try:
        res = evaluate_and_forecast(serie, horizon=horizon)
    except ValueError as exc:
        st.warning(f"No se puede pronosticar esta estacion: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE modelo", f"{res['mape_modelo']:.1f}%")
    c2.metric("RMSE modelo", f"{res['rmse_modelo']:.1f}")
    c3.metric("MAPE baseline (MM-7d)", f"{res['mape_baseline']:.1f}%")
    c4.metric("RMSE baseline", f"{res['rmse_baseline']:.1f}")

    fig = go.Figure()
    hist = pd.concat([res["train"].iloc[-120:], res["test"]])
    fig.add_scatter(x=hist.index, y=hist.values, name="Historico", line={"color": "#888"})
    fig.add_scatter(x=res["test"].index, y=res["pred_test"].values, name="Prediccion (holdout)", line={"color": "#1f77b4"})
    fig.add_scatter(x=res["test"].index, y=res["baseline_test"].values, name="Baseline MM-7d", line={"dash": "dot", "color": "#2ca02c"})
    fig.add_scatter(x=res["forecast"].index, y=res["forecast"].values, name=f"Pronostico +{horizon}d", line={"color": "#d62728"})
    fig.add_hline(y=ECA_PM25_24H, line_dash="dash", annotation_text="ECA 50 µg/m³")
    fig.update_layout(title=f"PM2.5 diario — {station} (Holt-Winters, estacionalidad semanal)", yaxis_title="µg/m³")
    st.plotly_chart(fig, use_container_width=True)

    if res["mape_modelo"] < res["mape_baseline"]:
        st.success("El modelo Holt-Winters supera a la media movil de 7 dias en el holdout.")
    else:
        st.warning("La media movil de 7 dias es competitiva: la serie es muy persistente en este periodo.")


def panel_crud(daily: pd.DataFrame, features: pd.DataFrame, results: dict) -> None:
    st.header("Panel 4 — Consultas: prediccion vs. valor real (API WAQI)")
    repo = get_repo()
    st.caption(
        f"Base de datos activa: **{repo.backend.upper()}**"
        + ("" if repo.backend == "supabase" else " (fallback local; configura SUPABASE_URL/ANON_KEY para usar Postgres)")
    )

    best = best_model_name(results)
    model = results["modelos"][best]["modelo"]

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
        valor_real = live["pm25_estimado_ugm3"] if live else None
        ok = repo.guardar(
            estacion=station,
            inputs={"fecha_features": str(rows["fecha_dia"].iloc[-1].date()), "modelo": best},
            valor_predicho=proba,
            valor_real=valor_real,
            fuente_en_vivo=live is not None,
        )
        c1, c2 = st.columns(2)
        c1.metric("Prob. de exceder ECA (modelo)", f"{proba:.1%}")
        if live:
            c2.metric(
                "PM2.5 en vivo (WAQI)",
                f"{live['pm25_estimado_ugm3']} µg/m³",
                help=f"AQI pm25={live['aqi_pm25']} medido {live['hora_medicion']} en {live['estacion_waqi']}",
            )
        else:
            c2.warning("API WAQI no respondio: se guardo solo la prediccion (fuente_en_vivo = false).")
        st.success("Consulta guardada." if ok else "No se pudo guardar la consulta.")

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
    st.title("🌫️ Calidad del aire en Lima Metropolitana")
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
        ["📊 EDA + Clustering", "🤖 Predictivo", "📈 Pronostico", "🗂️ Consultas (CRUD)"]
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
        results = cached_training(features, 0.2, 200, 0.1, "0.2-200-0.1")
        panel_crud(daily, features, results)


if __name__ == "__main__":
    main()
