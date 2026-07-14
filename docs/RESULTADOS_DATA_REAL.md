# Resultados con la data real de SENAMHI

**Fecha de ejecución:** 14 de julio de 2026
**Dataset:** CSV oficial "Monitoreo de los contaminantes del aire en Lima Metropolitana" (datosabiertos.gob.pe), descargado el 14/07/2026 (68.5 MB).

## 1. Calidad de datos (limpieza)

| Métrica | Valor |
|---|---|
| Registros horarios totales | 577,794 |
| Registros válidos (≥1 contaminante) | 464,638 (19.58% descartado) |
| Cobertura temporal | 2015-01-01 → 2024-05-31 |
| Estaciones | 7: Campo de Marte, Carabayllo, San Borja, San Juan de Lurigancho, San Martín de Porres, Santa Anita, Villa María del Triunfo |
| Nulos entre los válidos | PM10: 83,656 · PM2.5: 97,047 · NO2: 165,446 |
| Días-estación (≥12h de PM2.5) | 15,386 |

Nota de formato del CSV real: `HORA` viene como HHMMSS (ej. `50000` = 05:00) y las estaciones con guiones bajos (`CAMPO_DE_MARTE`); la limpieza lo maneja (`src/data/clean.py`).

## 2. Clustering (K-means, días por niveles de contaminación)

| k | Inercia | Silueta |
|---|---|---|
| **2** | 17,199.8 | **0.405** |
| 3 | 13,698.5 | 0.340 |
| 4 | 11,320.5 | 0.337 |
| 5-8 | ... | 0.325-0.276 |

k=2 sugerido por silueta. Perfiles: **cluster 0 "días críticos"** (PM10 99.3, PM2.5 36.4, NO2 30.4 µg/m³) vs **cluster 1 "días típicos"** (46.7 / 19.2 / 19.0).

## 3. Clasificación — `excede_pm25` (ECA 50 µg/m³, D.S. 003-2017-MINAM)

Dataset supervisado: 8,687 filas · clase "excede" = **3.6%** (fuerte desbalance → SMOTE aplicado solo en train). Features sin fuga: rezagos 1-3 días, media móvil 7d, calendario, estación.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.9522 | 0.3837 | **0.5238** | **0.4430** | **0.9386** |
| XGBoost | 0.9574 | 0.3962 | 0.3333 | 0.3621 | 0.9202 |

Matriz de confusión RF (test): TN=1622, FP=53, FN=30, TP=33.

**Conclusión:** Random Forest es el mejor modelo — con 3.6% de prevalencia, el recall de la clase "excede" es la métrica crítica (no alertar un día que sí excede cuesta más que una falsa alarma) y RF detecta el 52% de las excedencias vs 33% de XGBoost, con mejor F1 y AUC. La accuracy es engañosa por el desbalance (un modelo que nunca alerta tendría ~96%).

## 4. Pronóstico PM2.5 diario (Holt-Winters vs media móvil 7d, holdout 30 días)

| Estación | HW MAPE | HW RMSE | MM7d MAPE | MM7d RMSE | Gana |
|---|---|---|---|---|---|
| Campo de Marte | 31.6% | 15.0 | 24.3% | 10.2 | baseline |
| Carabayllo | 19.5% | 8.2 | 19.2% | 7.0 | baseline |
| San Borja | 28.1% | 13.4 | 18.9% | 8.1 | baseline |
| San Juan de Lurigancho | **19.8%** | 8.2 | 20.9% | 7.9 | **modelo** |
| San Martín de Porres | 32.7% | 9.9 | 21.7% | 6.6 | baseline |
| Santa Anita | 38.9% | 18.5 | 20.3% | 8.7 | baseline |
| Villa María del Triunfo | **20.3%** | **4.7** | 22.0% | 5.4 | **modelo** |

**Hallazgo honesto para el reporte:** el PM2.5 diario de Lima es altamente persistente; una media móvil de 7 días (actualizada con valores reales) es un baseline muy fuerte y Holt-Winters solo lo supera en 2 de 7 estaciones. Esto es un resultado válido y defendible: demuestra que se evaluó contra un baseline serio en lugar de reportar solo el modelo.

## Reproducir

```bash
python scripts/run_experiments.py   # requiere el CSV en data_cache/
```
