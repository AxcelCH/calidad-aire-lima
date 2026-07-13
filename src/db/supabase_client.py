"""Panel 4: repositorio del CRUD de consultas.

Backend primario: Supabase (Postgres). Si no hay credenciales configuradas o
Supabase falla, degrada a una base SQLite local (data_cache/consultas_local.db)
con la misma interfaz — la app nunca se cae por la base de datos.

Reglas de negocio 6:
- timestamp generado por el servidor, no editable
- editar solo permite modificar 'observacion'
- eliminar es borrado logico (columna 'eliminado')
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src.config import LOCAL_DB_PATH, SUPABASE_ANON_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

TABLE = "consultas"

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estacion TEXT NOT NULL,
    creado_en TEXT NOT NULL,
    inputs TEXT,
    valor_predicho REAL,
    valor_real REAL,
    fuente_en_vivo INTEGER NOT NULL DEFAULT 0,
    observacion TEXT DEFAULT '',
    eliminado INTEGER NOT NULL DEFAULT 0
);
"""


class ConsultasRepo:
    """Repositorio unico del CRUD. Detecta el backend disponible al crearse."""

    def __init__(self) -> None:
        self.backend = "sqlite"
        self._client = None
        if SUPABASE_URL and SUPABASE_ANON_KEY:
            try:
                from supabase import create_client

                self._client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                self.backend = "supabase"
            except Exception as exc:  # noqa: BLE001 - degradacion controlada
                logger.warning("Supabase no disponible, uso SQLite local: %s", exc)
        if self.backend == "sqlite":
            with self._sqlite() as conn:
                conn.execute(_SQLITE_SCHEMA)

    def _sqlite(self) -> sqlite3.Connection:
        return sqlite3.connect(LOCAL_DB_PATH)

    # ---------- CREATE ----------
    def guardar(
        self,
        estacion: str,
        inputs: dict,
        valor_predicho: float,
        valor_real: float | None,
        fuente_en_vivo: bool,
    ) -> bool:
        """Inserta una consulta. Devuelve True si se guardo bien."""
        row = {
            "estacion": estacion,
            "inputs": json.dumps(inputs, ensure_ascii=False),
            "valor_predicho": round(float(valor_predicho), 2),
            "valor_real": None if valor_real is None else round(float(valor_real), 2),
            "fuente_en_vivo": fuente_en_vivo,
            "observacion": "",
            "eliminado": False,
        }
        try:
            if self.backend == "supabase":
                self._client.table(TABLE).insert(row).execute()
            else:
                with self._sqlite() as conn:
                    conn.execute(
                        "INSERT INTO consultas (estacion, creado_en, inputs, valor_predicho,"
                        " valor_real, fuente_en_vivo, observacion, eliminado)"
                        " VALUES (?, ?, ?, ?, ?, ?, '', 0)",
                        (
                            row["estacion"],
                            datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            row["inputs"],
                            row["valor_predicho"],
                            row["valor_real"],
                            int(fuente_en_vivo),
                        ),
                    )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("No se pudo guardar la consulta: %s", exc)
            return False

    # ---------- READ ----------
    def listar(self, incluir_eliminados: bool = False) -> pd.DataFrame:
        """Consultas guardadas, mas recientes primero."""
        try:
            if self.backend == "supabase":
                query = self._client.table(TABLE).select("*").order("creado_en", desc=True)
                if not incluir_eliminados:
                    query = query.eq("eliminado", False)
                return pd.DataFrame(query.execute().data)
            with self._sqlite() as conn:
                where = "" if incluir_eliminados else "WHERE eliminado = 0"
                return pd.read_sql_query(
                    f"SELECT * FROM consultas {where} ORDER BY creado_en DESC", conn
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("No se pudo listar consultas: %s", exc)
            return pd.DataFrame()

    # ---------- UPDATE ----------
    def editar_observacion(self, consulta_id: int, observacion: str) -> bool:
        """Solo se puede editar la observacion (trazabilidad)."""
        try:
            if self.backend == "supabase":
                self._client.table(TABLE).update({"observacion": observacion}).eq(
                    "id", consulta_id
                ).execute()
            else:
                with self._sqlite() as conn:
                    conn.execute(
                        "UPDATE consultas SET observacion = ? WHERE id = ?",
                        (observacion, consulta_id),
                    )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("No se pudo editar la consulta %s: %s", consulta_id, exc)
            return False

    # ---------- DELETE (logico) ----------
    def eliminar(self, consulta_id: int) -> bool:
        """Borrado logico: marca eliminado=1, nunca DELETE fisico."""
        try:
            if self.backend == "supabase":
                self._client.table(TABLE).update({"eliminado": True}).eq(
                    "id", consulta_id
                ).execute()
            else:
                with self._sqlite() as conn:
                    conn.execute(
                        "UPDATE consultas SET eliminado = 1 WHERE id = ?", (consulta_id,)
                    )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("No se pudo eliminar la consulta %s: %s", consulta_id, exc)
            return False
