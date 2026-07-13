"""
Configuration loading for the DFT job-lifecycle tools.

Configuration is resolved with the following precedence (highest first):

    1. Environment variables  (MATCLAW_DFT_*)
    2. A YAML config file      (config.yaml, optionally under a `dft:` key)
    3. Built-in defaults

The YAML file is located via the ``MATCLAW_DFT_CONFIG`` environment variable,
or by searching the current working directory and the ``mcp/`` package root for
``config.yaml``. See ``config.example.yaml`` for the full schema.

The database backend is selected purely by ``database_url``:

    sqlite:////abs/path/dft_jobs.db          (default — no extra drivers needed)
    postgresql+psycopg://user:pw@host/db     (pip install "psycopg[binary]")
    mysql+pymysql://user:pw@host/db          (pip install pymysql)

Only the URL changes — the SQLAlchemy-backed persistence layer adapts
automatically.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

ENV_PREFIX = "MATCLAW_DFT_"


class SchedulerConfig(BaseModel):
    """HPC scheduler settings (engine-agnostic)."""

    type: str = "slurm"  # "slurm" | "local"
    submit_command: str = "sbatch"
    status_command: str = "squeue"
    cancel_command: str = "scancel"
    accounting_command: str = "sacct"
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    default_nodes: int = 1
    default_ntasks: int = 16
    default_walltime: str = "24:00:00"
    # Lines emitted verbatim into the batch script before the run commands,
    # e.g. ["module load vasp/6.4.2", "source ~/.orca_env"].
    modules: List[str] = Field(default_factory=list)
    prologue: List[str] = Field(default_factory=list)


class EngineConfig(BaseModel):
    """Per-engine execution settings.

    Command templates support ``${ntasks}``, ``${input_file}`` and
    ``${output_file}`` placeholders, substituted at submit time.
    """

    vasp_command: str = "mpirun -np ${ntasks} vasp_std"
    orca_command: str = "${orca_bin}/orca ${input_file} > ${output_file}"
    # Path exported as VASP_PP_PATH / PMG_VASP_PSP_DIR for POTCAR generation.
    vasp_pp_path: Optional[str] = None
    # Directory containing the `orca` and `orca_plot` binaries.
    orca_bin: Optional[str] = None


class DFTConfig(BaseModel):
    """Top-level DFT tool configuration."""

    workdir: str = "./dft_jobs"
    database_url: Optional[str] = None
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    engines: EngineConfig = Field(default_factory=EngineConfig)

    def resolved_workdir(self) -> Path:
        return Path(self.workdir).expanduser().resolve()

    def resolved_database_url(self) -> str:
        """Return the configured DB URL, defaulting to sqlite under ``workdir``."""
        if self.database_url:
            return self.database_url
        db_path = self.resolved_workdir() / "dft_jobs.db"
        return f"sqlite:///{db_path}"


def _find_config_file() -> Optional[Path]:
    explicit = os.environ.get(f"{ENV_PREFIX}CONFIG")
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path.cwd() / "config.yaml")
    # mcp/ package root (this file is mcp/tools/dft/config.py)
    candidates.append(Path(__file__).resolve().parents[2] / "config.yaml")
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def _load_yaml() -> Dict[str, Any]:
    path = _find_config_file()
    if not path:
        return {}
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    # Allow either a flat file or a nested `dft:` section.
    if "dft" in data and isinstance(data["dft"], dict):
        return data["dft"]
    return data


def _apply_env_overrides(cfg: DFTConfig) -> DFTConfig:
    env = os.environ

    def get(key: str) -> Optional[str]:
        return env.get(f"{ENV_PREFIX}{key}")

    if get("WORKDIR"):
        cfg.workdir = get("WORKDIR")  # type: ignore[assignment]
    if get("DATABASE_URL"):
        cfg.database_url = get("DATABASE_URL")
    if get("SCHEDULER"):
        cfg.scheduler.type = get("SCHEDULER")  # type: ignore[assignment]
    if get("ACCOUNT"):
        cfg.scheduler.account = get("ACCOUNT")
    if get("PARTITION"):
        cfg.scheduler.partition = get("PARTITION")
    if get("QOS"):
        cfg.scheduler.qos = get("QOS")
    if get("VASP_PP_PATH"):
        cfg.engines.vasp_pp_path = get("VASP_PP_PATH")
    if get("ORCA_BIN"):
        cfg.engines.orca_bin = get("ORCA_BIN")
    return cfg


@lru_cache(maxsize=1)
def get_config() -> DFTConfig:
    """Load and cache the DFT configuration (YAML overlaid with env vars)."""
    cfg = DFTConfig(**_load_yaml())
    return _apply_env_overrides(cfg)


def reset_config_cache() -> None:
    """Clear the cached config — primarily for tests that mutate the environment."""
    get_config.cache_clear()
