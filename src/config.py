"""Configuracion central del proyecto.

Lee las variables de entorno una sola vez (desde .env en local, o desde los
Secrets de Streamlit Community Cloud / Hugging Face Spaces en produccion,
donde no existe un archivo .env) y define rutas compartidas.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# Rutas donde Streamlit busca el secrets.toml (local del proyecto, home del usuario
# y el mount de Streamlit Community Cloud). Solo se consulta st.secrets si alguno
# existe: tocar st.secrets sin archivo emite un error a la UI *durante el import*,
# lo que ademas rompe el orden y hace fallar st.set_page_config().
_SECRETS_PATHS = (
    ROOT_DIR / ".streamlit" / "secrets.toml",
    Path.home() / ".streamlit" / "secrets.toml",
    Path("/mount/src") / ROOT_DIR.name / ".streamlit" / "secrets.toml",
)


def _secret_or_env(key: str, default: str = "") -> str:
    """os.getenv primero (.env local); si no hay valor, intenta st.secrets (Streamlit Cloud)."""
    value = os.getenv(key, "")
    if value:
        return value
    if not any(p.exists() for p in _SECRETS_PATHS):
        return default
    try:
        import streamlit as st

        return str(st.secrets.get(key, default))
    except Exception:
        return default


DEFAULT_SENAMHI_CSV_URL = (
    "https://www.datosabiertos.gob.pe/sites/default/files/"
    "Monitoreo%20de%20los%20contaminantes%20del%20aire%20en%20Lima%20Metropolitana"
    "%20-%20%5BServicio%20Nacional%20de%20Meteorolog%C3%ADa%20e%20Hidrolog%C3%ADa"
    "%20del%20Per%C3%BA%20-%20SENAMHI%5D_1.csv"
)

# Espejo del CSV en un Release de este repo. El portal gob.pe bloquea las IPs
# de proveedores cloud (403 desde Streamlit Cloud/GCP), asi que se usa como
# fuente de respaldo cuando la descarga oficial falla. Mismo archivo, sha256
# 896ae8a1...c40d4 verificado.
SENAMHI_CSV_MIRROR_URL = (
    "https://github.com/AxcelCH/calidad-aire-lima/releases/download/"
    "data-senamhi-v1/senamhi_aire_lima_20260714.csv"
)

WAQI_API_TOKEN: str = _secret_or_env("WAQI_API_TOKEN")
SUPABASE_URL: str = _secret_or_env("SUPABASE_URL")
SUPABASE_ANON_KEY: str = _secret_or_env("SUPABASE_ANON_KEY")
SENAMHI_CSV_URL: str = _secret_or_env("SENAMHI_CSV_URL", DEFAULT_SENAMHI_CSV_URL)
APP_ENV: str = _secret_or_env("APP_ENV", "development")

DATA_CACHE_DIR = ROOT_DIR / "data_cache"
MODELS_CACHE_DIR = ROOT_DIR / "models_cache"
LOCAL_DB_PATH = DATA_CACHE_DIR / "consultas_local.db"

DATA_CACHE_DIR.mkdir(exist_ok=True)
MODELS_CACHE_DIR.mkdir(exist_ok=True)
