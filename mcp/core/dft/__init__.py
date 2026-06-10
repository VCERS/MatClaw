"""
core.dft — the DFT job-execution subsystem.

This package holds the engine-agnostic infrastructure behind the DFT MCP tools:

    config     : layered YAML + environment configuration
    models     : the persisted JobRecord and lifecycle states
    store      : SQLAlchemy-backed job persistence (sqlite default; any SQL DB)
    scheduler  : SLURM / local control plane (submit, status, cancel)
    engines    : VASP and ORCA adapters (input generation + result parsing)

The thin MCP tool functions live separately under ``tools.dft`` and orchestrate
these components. Nothing here is exposed to agents directly.
"""

from .config import DFTConfig, get_config, reset_config_cache
from .engines import get_engine, supported_engines
from .models import JobRecord, JobState
from .scheduler import get_scheduler
from .store import get_job_store, reset_store_cache

__all__ = [
    "DFTConfig",
    "get_config",
    "reset_config_cache",
    "get_engine",
    "supported_engines",
    "JobRecord",
    "JobState",
    "get_scheduler",
    "get_job_store",
    "reset_store_cache",
]
