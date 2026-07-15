# Contexto del Proyecto — Bitácora para IA

> **Instrucciones de uso:** pega este archivo completo (junto con `ARQUITECTURA_Y_DISENO.md` y, si hace falta, la propuesta en `Propuesta/Propuesta_Proyecto_MineriaDeDatos_v2.docx`) al inicio de una conversación nueva con cualquier modelo de IA (Claude, Fable u otro) para que continúe el proyecto sin que tengas que reexplicar todo desde cero. Actualiza este archivo después de cada sesión de trabajo relevante — es más importante mantenerlo al día que completo.

---

## Estado actual (una línea)

Auditoría contra la rúbrica del PDF completada (14/07): se agregaron DBSCAN (Panel 1), MLP como 3er modelo con epochs/activation en vivo y métricas por clase (Panel 2), línea de tendencia (Panel 3) y **entrada manual de datos en el Panel 4**; checklist punto por punto en `docs/CHECKLIST_REQUISITOS.md`; hosting sigue pendiente (candidato: Streamlit Community Cloud).

## Resumen del proyecto

- **Tema:** clasificación de excedencia del Estándar de Calidad Ambiental (ECA) para PM2.5/PM10, clustering de estaciones, pronóstico de contaminantes, y CRUD de consultas, para Lima Metropolitana.
- **Dataset histórico:** CSV oficial de SENAMHI en datosabiertos.gob.pe (verificado, descarga directa).
- **Dato en vivo:** API WAQI (aqicn.org/api.waqi.info), verificada con token real el 13/07/2026 — 8 estaciones activas en Lima.
- **Arquitectura:** app única en Streamlit, hosteada en Hugging Face Spaces, con Supabase (Postgres) como base de datos del CRUD.
- **Equipo:** grupo de 3 integrantes (nombres pendientes de completar).

## Decisiones clave tomadas (registro tipo ADR)

| # | Decisión | Alternativas descartadas | Motivo |
|---|---|---|---|
| 1 | Tema: calidad del aire (PM10/PM2.5/NO2) en Lima Metropolitana | Accidentes viales, deserción universitaria, turismo, microcréditos | Dataset gubernamental verificado, normativa clara (ECA), bajo riesgo legal |
| 2 | Dataset histórico: CSV oficial SENAMHI vía datosabiertos.gob.pe | Kaggle/UCI (prohibido por la rúbrica) | Fuente peruana verificable, descarga directa sin solicitar cita |
| 3 | Dato en vivo: API WAQI (aqicn.org) | Scraping directo de Google Maps (reseñas de tiendas de PC) | API oficial gratuita, sin riesgo de violar Términos de Servicio; ya probada con token real |
| 4 | Arquitectura: Streamlit todo-en-uno (frontend = backend) | Frontend en Vercel/Firebase + backend FastAPI separado | El equipo tiene nivel básico-intermedio; menos piezas = menos riesgo el día de la demo |
| 5 | Hosting: Hugging Face Spaces | Streamlit Community Cloud, Render, Railway | Pensado para apps de ML, sin timeouts agresivos, integra bien con GitHub |
| 6 | Base de datos del CRUD: Supabase (Postgres) | Google Sheets (gspread), Firestore | Balance entre tener una "app real" con Postgres/REST y no complicar demasiado la configuración |
| 7 | Agregación temporal para modelos: promedio diario, no horario | Usar el dato horario crudo | El ECA se define como promedio de 24h; mantiene coherencia entre la etiqueta y la norma peruana |
| 8 | Repositorio público en la cuenta GitHub **AxcelCH** (`calidad-aire-lima`) | Cuenta de equipo | Definido al momento de publicar (14/07/2026); 10 commits convencionales |
| 9 | Variable objetivo del Panel 2: **PM2.5** (`excede_pm25`) | PM10 | Contaminante crítico de Lima, decisión confirmada por Jeremi |
| 10 | CRUD con **fallback SQLite local** si Supabase no está configurado | Requerir Supabase obligatorio | Degradación controlada: la app nunca se cae por la BD; Supabase se conecta después solo con .env |
| 11 | **Hosting pospuesto** — el free tier de HF Spaces ya no soporta Streamlit (solo ZeroGPU/Gradio; CPU Basic requiere PRO, verificado 14/07/2026) | Reescribir la app a Gradio; Streamlit Community Cloud | Decisión de Jeremi: entregar el repo; candidato principal cuando se retome: Streamlit Community Cloud (gratis, deploy directo del repo GitHub) |

## Próximos pasos inmediatos

1. ~~Descargar el CSV real~~ ✅ Hecho (14/07): CSV en `data_cache/`, modelos ejecutados, métricas en `docs/RESULTADOS_DATA_REAL.md`.
2. Usar las métricas reales en el Reporte PDF (tablas listas en `docs/RESULTADOS_DATA_REAL.md`).
3. Decidir el hosting definitivo (candidato: Streamlit Community Cloud con login GitHub de AxcelCH; alternativa: HF Spaces PRO o app reescrita en Gradio).
4. Crear el proyecto en Supabase y cargar `SUPABASE_URL`/`SUPABASE_ANON_KEY` en `.env` (mientras tanto el CRUD usa SQLite local automáticamente).
5. Repartir los 4 paneles entre los 3 integrantes y ensayar las preguntas de modificación en vivo (hiperparámetros expuestos: `test_size`, `n_estimators`, `learning_rate`, `k`).
6. Presentar la propuesta al docente en S15 (aún no aprobada formalmente).

## Preguntas abiertas (sin resolver aún)

- ¿Cómo se reparten los 4 paneles entre los 3 integrantes?
- ¿Hosting definitivo? (ver decisión 11: HF free ya no sirve para Streamlit)
- ¿Quién crea y administra el proyecto de Supabase?

**Resueltas:** variable objetivo = PM2.5 (decisión 9); repo en GitHub de AxcelCH (decisión 8).

## Archivos relevantes del proyecto

| Archivo | Contenido | Carpeta |
|---|---|---|
| `Propuesta_Proyecto_MineriaDeDatos_v2.docx` | Propuesta formal de tema (para entrega S15), incluye fuente de datos, mapeo a paneles, objetivos SMART, KPIs, prueba de la API WAQI | `Propuesta/` |
| `ARQUITECTURA_Y_DISENO.md` | Documento técnico completo: stack, arquitectura, reglas de negocio, requerimientos funcionales, buenas prácticas, plan de despliegue | `Implementacion/` |
| `CONTEXT_LOG.md` | Este archivo — bitácora de contexto para retomar el proyecto con cualquier IA | `Implementacion/` |
| `.env.example` | Plantilla de variables de entorno necesarias (sin valores reales) | `Implementacion/` |
| `.gitignore` | Excluye `.env`, cachés y artefactos pesados del repositorio | `Implementacion/` |
| `calidad-aire-lima/` | **Código fuente completo del proyecto** (app.py, src/, tests/, scripts/, docs/) — espejo del repo público | raíz del proyecto |

## Enlaces del proyecto

- **Repo público (entregable):** https://github.com/AxcelCH/calidad-aire-lima
- **Space HF (creado pero no funcional** — free tier sin soporte Streamlit; ver decisión 11): https://huggingface.co/spaces/Lecxa/calidad-aire-lima
- **Dataset oficial:** https://www.datosabiertos.gob.pe/dataset/monitoreo-de-los-contaminantes-del-aire-en-lima-metropolitana-servicio-nacional-de
- El token WAQI está solo en `calidad-aire-lima/.env` (local, gitignored) — nunca subirlo al repo.

## Resultados con la DATA REAL de SENAMHI (14/07/2026)

CSV oficial descargado (68.5 MB, 577,794 registros horarios, 2015→2024-05, 7 estaciones). Resultados completos en `calidad-aire-lima/docs/RESULTADOS_DATA_REAL.md`. Resumen: 19.58% de registros descartados (todos los contaminantes vacíos); clustering k=2 (silueta 0.405, perfiles "días críticos" vs "típicos"); clasificación `excede_pm25` con 3.6% de prevalencia → SMOTE, **Random Forest gana** (F1 0.443, recall 0.524, AUC 0.939 vs XGBoost F1 0.362); pronóstico: la media móvil 7d es un baseline fuerte, Holt-Winters solo gana en SJL y VMT (hallazgo honesto para el reporte). Ajustes de código por el formato real: HORA en HHMMSS, estaciones con guiones bajos, lectura con motor C (~10x más rápida), serie de pronóstico limitada a ~3 años recientes con frecuencia diaria continua. Verificación previa: CRUD OK (SQLite), AQI→µg/m³ validada contra San Borja en vivo, 8 tests pytest pasan.

## Cómo continuar esta conversación con otra IA

1. Comparte este archivo completo.
2. Si la tarea es de arquitectura/decisiones técnicas, comparte también `ARQUITECTURA_Y_DISENO.md`.
3. Si la tarea es sobre el contenido académico/entrega, comparte también la propuesta en Word.
4. Pide explícitamente que, al terminar la sesión, actualice las secciones "Estado actual", "Próximos pasos inmediatos", "Preguntas abiertas" y agregue una fila en "Historial de sesiones" de este archivo (y la tabla de la sección 15 de `ARQUITECTURA_Y_DISENO.md` si hubo cambios de arquitectura).

## Historial de sesiones

| Fecha | Qué se hizo | Archivos creados/modificados |
|---|---|---|
| 2026-07-12 | Elegido el tema (calidad del aire, SENAMHI) y creada la propuesta inicial | `Propuesta_Proyecto_MineriaDeDatos.docx` |
| 2026-07-13 | Explorada alternativa de recolección por API/scraping; descartado scraping de Google Maps por riesgo de ToS; validada la API WAQI con un token real (San Borja y 7 estaciones más); propuesta actualizada | `Propuesta_Proyecto_MineriaDeDatos_v2.docx` |
| 2026-07-13 | Definida la arquitectura de implementación (Streamlit + Hugging Face Spaces + Supabase) y creada toda la documentación técnica | `ARQUITECTURA_Y_DISENO.md`, `CONTEXT_LOG.md`, `.env.example`, `.gitignore` |
| 2026-07-14 | **Implementación completa del proyecto**: código de los 4 paneles + modelos + CRUD + tests; repo público creado y poblado (github.com/AxcelCH/calidad-aire-lima, 10 commits); token WAQI verificado en vivo y sacado del `.env.example` público; pipeline verificado end-to-end con datos sintéticos (CSV real pendiente) | `app.py`, `src/`, `tests/`, `scripts/` |
| 2026-07-14 | **Auditoría contra la rúbrica del PDF + brechas cerradas**: DBSCAN con comparación de silueta y detección de outliers (Panel 1); MLP como 3er modelo (tema 8 pide MLP vs RF vs XGBoost) con sliders de epochs/activation para la pregunta en vivo de redes neuronales; métricas por clase bajo las matrices de confusión; línea de tendencia MM-30d (Panel 3); **entrada manual de datos en el Panel 4** (modo manual del formulario: el usuario ingresa PM2.5/PM10/NO₂ de días previos + fecha, prellenado con el último dato histórico); 6 tests nuevos (anti-leakage de features y DBSCAN); checklist de cumplimiento | `app.py`, `src/models/classifier.py`, `src/models/clustering.py`, `tests/test_features_y_clustering.py`, `docs/CHECKLIST_REQUISITOS.md`, `README.md` |