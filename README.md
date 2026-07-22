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

**Estado del despliegue:** hosting en [Streamlit Community Cloud](https://share.streamlit.io) (gratis, deploy directo desde GitHub). Nota: el free tier de HF Spaces ya no soporta el SDK Streamlit (jul-2026), por eso se descartó.

## Los 4 paneles

| Panel | Contenido | Técnicas |
|---|---|---|
| 📊 EDA + Clustering | Estadísticas, histogramas/boxplots con outliers (1.5·IQR), correlaciones, mapa de estaciones, K-means con codo y silueta, DBSCAN con detección de outliers | K-means vs DBSCAN, IQR |
| 🤖 Predictivo | ¿El promedio diario de PM2.5 excederá el ECA peruano (50 µg/m³, D.S. 003-2017-MINAM)? Matriz de confusión y métricas por clase | MLP vs Random Forest vs XGBoost, SMOTE, SHAP |
| 📈 Pronóstico | Serie diaria de PM2.5 por estación con tendencia, pronóstico a 4-14 días con MAPE/RMSE | Holt-Winters vs ARIMA(1,1,1) vs baseline MM-7d |
| 🗂️ CRUD | Guarda consultas (automáticas con WAQI en vivo, o con **entrada manual de datos**): predicción del modelo + datos de entrada. Editar/eliminar con trazabilidad | Supabase (Postgres) con fallback SQLite |

La correspondencia punto por punto con la rúbrica del curso está en [`docs/CHECKLIST_REQUISITOS.md`](docs/CHECKLIST_REQUISITOS.md).

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

## Despliegue en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con la cuenta de GitHub dueña del repo (`AxcelCH`).
2. **New app** → selecciona el repo `AxcelCH/calidad-aire-lima` → **branch: `angela`** → **main file: `app.py`**.
3. En *Advanced settings*, confirma la versión de Python 3.11 (el repo ya incluye [`runtime.txt`](runtime.txt) para fijarla; `pandas`/`numpy` en las versiones pineadas no tienen wheels para Python 3.13+).
4. En *Secrets*, pega en formato TOML los mismos valores de tu `.env` local:

   ```toml
   WAQI_API_TOKEN = "..."
   SUPABASE_URL = "..."
   SUPABASE_ANON_KEY = "..."
   ```

   Sin esto, el CRUD sigue funcionando con SQLite local y el Panel 4 sin dato en vivo (degradación controlada).
5. **Deploy**. La primera corrida descarga el CSV de SENAMHI (~65 MB), tarda más que las siguientes.

## Equipo

Grupo de 3 integrantes — UNMSM FISI, Minería de Datos 2026-I.
