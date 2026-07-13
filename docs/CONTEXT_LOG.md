# Contexto del Proyecto — Bitácora para IA

> **Instrucciones de uso:** pega este archivo completo (junto con `ARQUITECTURA_Y_DISENO.md` y, si hace falta, la propuesta en `Propuesta/Propuesta_Proyecto_MineriaDeDatos_v2.docx`) al inicio de una conversación nueva con cualquier modelo de IA (Claude, Fable u otro) para que continúe el proyecto sin que tengas que reexplicar todo desde cero. Actualiza este archivo después de cada sesión de trabajo relevante — es más importante mantenerlo al día que completo.

---

## Estado actual (una línea)

Proyecto de minería de datos (UNMSM, curso Minería de Datos 2026-I) sobre calidad del aire en Lima Metropolitana. Tema y arquitectura definidos; implementación (código) todavía no iniciada.

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

## Próximos pasos inmediatos

1. Definir nombres del grupo, integrantes y repositorio de GitHub (pendiente en la ficha de la propuesta).
2. Resolver las preguntas abiertas de la sección 14 de `ARQUITECTURA_Y_DISENO.md` (PM2.5 vs PM10 como variable objetivo, reparto de paneles entre integrantes, quién administra las cuentas de Hugging Face/Supabase).
3. Crear el repositorio con la estructura de carpetas de la sección 5 del documento de arquitectura.
4. Descargar el CSV de SENAMHI y hacer un EDA exploratorio rápido (confirmar cobertura temporal y estaciones activas) antes de comprometerse con el enfoque final.
5. Registrar el token de la API WAQI (ya se probó uno personal, pero cada integrante debería tener el suyo para desarrollo) y crear el proyecto en Supabase.
6. Presentar la propuesta al docente en S15 (aún no aprobada formalmente).

## Preguntas abiertas (sin resolver aún)

- ¿PM2.5 o PM10 como variable objetivo principal de clasificación?
- ¿Cómo se reparten los 4 paneles entre los 3 integrantes?
- ¿Quién administra las cuentas de Hugging Face Spaces y Supabase?
- ¿A nombre de quién queda el repositorio de GitHub?

## Archivos relevantes del proyecto

| Archivo | Contenido | Carpeta |
|---|---|---|
| `Propuesta_Proyecto_MineriaDeDatos_v2.docx` | Propuesta formal de tema (para entrega S15), incluye fuente de datos, mapeo a paneles, objetivos SMART, KPIs, prueba de la API WAQI | `Propuesta/` |
| `ARQUITECTURA_Y_DISENO.md` | Documento técnico completo: stack, arquitectura, reglas de negocio, requerimientos funcionales, buenas prácticas, plan de despliegue | `Implementacion/` |
| `CONTEXT_LOG.md` | Este archivo — bitácora de contexto para retomar el proyecto con cualquier IA | `Implementacion/` |
| `.env.example` | Plantilla de variables de entorno necesarias (sin valores reales) | `Implementacion/` |
| `.gitignore` | Excluye `.env`, cachés y artefactos pesados del repositorio | `Implementacion/` |

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
