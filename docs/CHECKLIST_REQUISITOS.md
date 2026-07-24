# Checklist de requisitos — TrabajoFinal_MineriadeDatos_2026I.pdf

Auditoría del proyecto contra la rúbrica del curso (14/07/2026). Tema elegido: **#8 — Contaminación del aire en Lima (PM2.5, NO₂)** con datos de SENAMHI.

## Panel 1 — EDA + Clustering

| Requisito del PDF | Estado | Dónde |
|---|---|---|
| Estadísticas descriptivas | ✅ | Tabla mean/median/std por estación |
| Histogramas | ✅ | Histograma por contaminante (selector) |
| Mapa de correlación | ✅ | Heatmap PM10/PM2.5/NO₂ |
| Boxplots | ✅ | Boxplot por estación |
| Outliers (1.5·IQR) | ✅ | Tabla con Q1/Q3/IQR/límites/% outliers |
| K-means con método del codo | ✅ | Gráfico de inercia k=2..8 |
| Coeficiente de silueta | ✅ | Gráfico de silueta k=2..8 + k ajustable en vivo |
| Visualización de clusters | ✅ | Scatter PM2.5 vs PM10 + perfiles promedio |
| DBSCAN (comparar silueta, detectar outliers) | ✅ | Sección DBSCAN con eps/min_samples en vivo, silueta comparada vs K-means |
| Extras del tema 8: EDA temporal | ✅ | Excedencias ECA por mes, heatmap hora×día de semana, mapa geográfico |

## Panel 2 — Predictivo

| Requisito del PDF | Estado | Dónde |
|---|---|---|
| ≥ 2 modelos comparados | ✅ | **MLP vs Random Forest vs XGBoost** (los 3 que sugiere el tema 8) |
| Matriz de confusión | ✅ | Una por modelo, con colores |
| Precisión, recall, F1, ROC-AUC | ✅ | Tabla global + **métricas por clase** (expander) + accuracy |
| Justificación escrita de la elección | ✅ | Mensaje con F1/ROC-AUC y análisis de costo FP vs FN |
| SHAP summary plot (global) | ✅ | TreeExplainer sobre el mejor modelo de árboles |
| SHAP force plot (local, 1 instancia) | ✅ | Selector de fila del test |
| SMOTE / class_weight si desbalance > 80/20 | ✅ | SMOTE automático solo sobre train si minoritaria < 20 % |
| Curva ROC | ✅ | Comparación de los 3 modelos |
| Hiperparámetros modificables en vivo | ✅ | test_size, n_estimators, learning_rate, epochs (MLP), activation (MLP) |

## Panel 3 — Pronóstico

| Requisito del PDF | Estado | Dónde |
|---|---|---|
| Serie temporal graficada con tendencia | ✅ | Histórico + media móvil 30d como tendencia |
| Pronóstico ≥ 4 períodos | ✅ | Horizonte ajustable 4–14 días |
| MAPE y RMSE visibles en el panel | ✅ | Métricas de Holt-Winters, ARIMA y baseline |
| Modelo válido (MM / suav. exp. / ARIMA / Prophet) | ✅ | Holt-Winters + ARIMA(1,1,1) + baseline media móvil 7d |
| Comparación con baseline | ✅ | Conclusión automática del mejor modelo en holdout |

## Panel 4 — CRUD de consultas

| Requisito del PDF | Estado | Dónde |
|---|---|---|
| Formulario para guardar consulta (datos de entrada + predicción devuelta) | ✅ | Dos modos: **automático** (último día histórico + WAQI en vivo) y **manual** (el usuario ingresa PM2.5/PM10/NO₂ de días previos y la fecha) |
| Lista de consultas guardadas | ✅ | Tabla + exportar CSV |
| Botón editar | ✅ | Edición de observación (trazabilidad) |
| Botón eliminar | ✅ | Borrado lógico |
| Timestamp automático | ✅ | Generado por el servidor (UTC) |
| CRUD funcional desde el navegador | ✅ | Supabase (Postgres) con fallback SQLite |

## Requisitos transversales

| Requisito | Estado | Notas |
|---|---|---|
| Dataset peruano verificable (no Kaggle/UCI) | ✅ | SENAMHI vía datosabiertos.gob.pe (ODC-BY) + API WAQI en vivo |
| Dashboard en línea | ⏳ | Pendiente de hosting (candidato: Streamlit Community Cloud); corre local |
| Repo GitHub con código + README | ✅ | github.com/AxcelCH/calidad-aire-lima |
| Gráficos interactivos | ✅ | Plotly en todos los paneles |
| Código modificable en vivo | ✅ | 7 hiperparámetros expuestos como sliders/selects |
| Reporte PDF (SMART, KPIs, narrativa, justificación) | ⏳ | Entregable aparte; métricas listas en `docs/RESULTADOS_DATA_REAL.md` |
| Tests | ✅ | `tests/` — reglas ECA, limpieza, anti-leakage de features, DBSCAN |

## Preparación para preguntas de modificación en vivo (sección 5 del PDF)

| Pregunta típica | Cubierta por |
|---|---|
| Cambiar train/test 70/30 → 90/10 | Slider `test_size` |
| Subir n_estimators de 100 a 500 | Slider `n_estimators` |
| Bajar learning_rate de XGBoost | Slider `learning_rate` |
| Épocas 10→50 y ReLU→sigmoide en la red neuronal | Slider `epochs (MLP)` + select `activation` (logistic = sigmoide) |
| Cambiar k de 3 a 6 en K-means | Slider `k` + codo y silueta ya graficados |
| ¿MAPE aceptable? ¿Qué modelo probarías? | Panel 3 compara HW vs ARIMA vs baseline |
| ¿Recall bajo en clase positiva? | SMOTE automático + métricas por clase visibles |
| ¿Por qué RF sobre un modelo más simple? | Tabla comparativa + curvas ROC + mensaje de justificación |
