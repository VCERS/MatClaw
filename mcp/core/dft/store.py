"""
Persistence layer for DFT job records.

A single SQLAlchemy Core backend serves every supported database; the engine is
chosen entirely by the ``database_url`` in the configuration:

    sqlite:////abs/path/dft_jobs.db          (default — bundled with Python)
    postgresql+psycopg://user:pw@host/db     (pip install "psycopg[binary]")
    mysql+pymysql://user:pw@host/db          (pip install pymysql)

JSON-typed columns (``sqlalchemy.JSON``) store the dict/list fields portably
across all three backends. The job store is the durable hand-off between the
short, stateless MCP tool calls — a ``job_id`` written now is read back hours
later, possibly by a different worker process or a fresh agent session.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.engine import Engine as SAEngine

from .config import DFTConfig, get_config
from .models import JSON_FIELDS, JobRecord, _now

_metadata = MetaData()

# One flat table; engine-specific payloads live in the JSON columns.
dft_jobs = Table(
    "dft_jobs",
    _metadata,
    Column("job_id", String(64), primary_key=True),
    Column("engine", String(32), nullable=False),
    Column("calc_type", String(64), nullable=False),
    Column("state", String(32), nullable=False, index=True),
    Column("workdir", String(1024), nullable=False),
    Column("label", String(256)),
    Column("scheduler_id", String(128), index=True),
    Column("parent_job_id", String(64)),
    Column("created_at", String(64)),
    Column("updated_at", String(64)),
    Column("submitted_at", String(64)),
    Column("completed_at", String(64)),
    Column("resolved_params", JSON, default=dict),
    Column("resources", JSON, default=dict),
    Column("warnings", JSON, default=list),
    Column("results", JSON, default=dict),
    Column("message", String(2048)),
)

_COLUMNS = [c.name for c in dft_jobs.columns]


class JobStore:
    """Thin CRUD wrapper over the ``dft_jobs`` table."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        connect_args = {}
        if database_url.startswith("sqlite"):
            # Allow use across threads/workers; ensure the parent dir exists.
            connect_args = {"check_same_thread": False}
            db_path = database_url.replace("sqlite:///", "", 1)
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: SAEngine = create_engine(
            database_url, future=True, connect_args=connect_args
        )
        if database_url.startswith("sqlite"):
            # WAL keeps concurrent readers/writers from blocking each other.
            with self.engine.begin() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        _metadata.create_all(self.engine)

    # -- CRUD ----------------------------------------------------------------

    def create(self, record: JobRecord) -> JobRecord:
        with self.engine.begin() as conn:
            conn.execute(dft_jobs.insert().values(**self._to_row(record)))
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(dft_jobs).where(dft_jobs.c.job_id == job_id)
            ).mappings().first()
        return self._from_row(row) if row else None

    def update(self, job_id: str, **fields) -> Optional[JobRecord]:
        """Patch a subset of columns. Unknown keys are ignored."""
        patch = {k: v for k, v in fields.items() if k in _COLUMNS and k != "job_id"}
        patch["updated_at"] = _now()
        with self.engine.begin() as conn:
            conn.execute(
                update(dft_jobs).where(dft_jobs.c.job_id == job_id).values(**patch)
            )
        return self.get(job_id)

    def list(
        self,
        state: Optional[str] = None,
        engine: Optional[str] = None,
        limit: int = 100,
    ) -> List[JobRecord]:
        stmt = select(dft_jobs)
        if state:
            stmt = stmt.where(dft_jobs.c.state == state)
        if engine:
            stmt = stmt.where(dft_jobs.c.engine == engine)
        stmt = stmt.order_by(dft_jobs.c.created_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._from_row(r) for r in rows]

    def delete(self, job_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(dft_jobs).where(dft_jobs.c.job_id == job_id)
            )
        return bool(result.rowcount)

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(dft_jobs)).scalar() or 0)

    # -- (de)serialisation ---------------------------------------------------

    @staticmethod
    def _to_row(record: JobRecord) -> dict:
        # JSON columns accept native dict/list objects on every backend.
        return record.to_dict()

    @staticmethod
    def _from_row(row) -> JobRecord:
        data = dict(row)
        # JSON columns round-trip as native objects, but guard against NULLs.
        for key in JSON_FIELDS:
            if data.get(key) is None:
                data[key] = [] if key == "warnings" else {}
        return JobRecord(**data)


@lru_cache(maxsize=8)
def _store_for_url(database_url: str) -> JobStore:
    store = JobStore(database_url)
    store.init_schema()
    return store


def get_job_store(config: Optional[DFTConfig] = None) -> JobStore:
    """Return a process-cached job store for the configured database URL."""
    cfg = config or get_config()
    return _store_for_url(cfg.resolved_database_url())


def reset_store_cache() -> None:
    """Clear the cached job stores — used by tests that switch databases."""
    _store_for_url.cache_clear()
