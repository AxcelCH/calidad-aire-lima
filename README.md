---
title: Calidad del Aire Lima
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
license: mit
---

# 🌫️ Predicción y monitoreo de la calidad del aire en Lima Metropolitana

Proyecto del curso **Minería de Datos 2026-I (UNMSM — FISI)**. Dashboard analítico de 4 paneles construido con **datos reales del estado peruano**: el histórico horario de contaminantes de **SENAMHI** ([datosabiertos.gob.pe](https://www.datosabiertos.gob.pe/dataset/monitoreo-de-los-contaminantes-del-aire-en-lima-metropolitana-servicio-nacional-de), licencia ODC-BY) y el dato en vivo de la **API WAQI** ([aqicn.org](https://aqicn.org)).

**Estado del despliegue:** hosting en definición. La app corre local con `streamlit run app.py`; el candidato para hosting gratuito es [Streamlit Community Cloud](https://share.streamlit.io) (conectar este repo → main file `app.py` → cargar secrets). Nota: el free tier de HF Spaces ya no soporta el SDK Streamlit (jul-2026).

## Los 4 paneles

| Panel | Contenido | Técnicas |
|---|---|---|
| 📊 EDA + Clustering | Estadísticas, histogramas/boxplots con outliers, correlaciones, K-means con codo y silueta | K-means, IQR |
| 🤖 Predictivo | ¿El promedio diario de PM2.5 excederá el ECA peruano (50 µg/m³, D.S. 003-2017-MINAM)? | Random Forest vs XGBoost, SMOTE, SHAP |
| 📈 Pronóstico | Serie diaria de PM2.5 por estación, pronóstico a 4-14 días con MAPE/RMSE vs baseline | Holt-Winters (statsmodels) |
| 🗂️ CRUD | Guarda consultas: predicción del modelo vs valor real en vivo (WAQI). Editar/eliminar con trazabilidad | Supabase (Postgres) con fallback SQLite |

## Arranque local

```bash
git clone <este-repo>
cd calidad-aire-lima
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # completa WAQI_API_TOKEN (y Supabase si tienes)
streamlit run app.py
```

La primera corrida descarga el CSV oficial de SENAMHI (~65 MB) a `data_cache/`. Si ya lo tienes descargado, colócalo ahí como `senamhi_aire_lima.csv` y no tocará la red.

### Tests

```bash
pytest tests/ -v
```

### Comparación de modelos por consola (sin UI)

```bash
python scripts/run_experiments.py
```

## Arquitectura

Un solo servicio: **Streamlit es frontend y backend a la vez** (KISS). Hosting en Hugging Face Spaces, base de datos Supabase con degradación controlada a SQLite local. Detalles completos en [`docs/ARQUITECTURA_Y_DISENO.md`](docs/ARQUITECTURA_Y_DISENO.md).

```
Fuentes (SENAMHI CSV + API WAQI) → src/ingest → src/data (limpieza + reglas ECA)
    → src/models (K-means | RF vs XGBoost + SHAP | Holt-Winters) → app.py (4 tabs)
    → src/db (Supabase/SQLite, CRUD Panel 4)
```

## Reglas de negocio clave

- **Excedencia ECA:** PM2.5 > 50 µg/m³ o PM10 > 100 µg/m³ en promedio de 24 h.
- **Agregación diaria** por estación (el ECA se define a 24 h; sin fuga de datos: las features del clasificador solo usan días anteriores).
- **SMOTE** solo sobre entrenamiento si la clase minoritaria < 20 %.
- **CRUD trazable:** timestamp de servidor, edición solo de observación, borrado lógico.
- **Degradación controlada:** si WAQI o Supabase fallan, la app sigue funcionando.

## Variables de entorno

Ver [`.env.example`](.env.example). En Hugging Face Spaces se cargan en *Settings → Variables and secrets* con los mismos nombres.

## Equipo

Grupo de 3 integrantes — UNMSM FISI, Minería de Datos 2026-I.
