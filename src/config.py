"""Configuracion central del proyecto.

Lee las variables de entorno una sola vez (desde .env en local o desde los
Secrets de Hugging Face Spaces en produccion) y define rutas compartidas.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DEFAULT_SENAMHI_CSV_URL = (
    "https://www.datosabiertos.gob.pe/sites/default/files/"
    "Monitoreo%20de%20los%20contaminantes%20del%20aire%20en%20Lima%20Metropolitana"
    "%20-%20%5BServicio%20Nacional%20de%20Meteorolog%C3%ADa%20e%20Hidrolog%C3%ADa"
    "%20del%20Per%C3%BA%20-%20SENAMHI%5D_1.csv"
)

WAQI_API_TOKEN: str = os.getenv("WAQI_API_TOKEN", "")
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
SENAMHI_CSV_URL: str = os.getenv("SENAMHI_CSV_URL", DEFAULT_SENAMHI_CSV_URL)
APP_ENV: str = os.getenv("APP_ENV", "development")

DATA_CACHE_DIR = ROOT_DIR / "data_cache"
MODELS_CACHE_DIR = ROOT_DIR / "models_cache"
LOCAL_DB_PATH = DATA_CACHE_DIR / "consultas_local.db"

DATA_CACHE_DIR.mkdir(exist_ok=True)
MODELS_CACHE_DIR.mkdir(exist_ok=True)
