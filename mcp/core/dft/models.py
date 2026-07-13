"""
Data model for persisted DFT jobs.

A :class:`JobRecord` is the single source of truth for one calculation as it
moves through its lifecycle (prepared -> submitted -> running -> completed /
failed / cancelled). It is intentionally engine-agnostic; engine-specific
details live in the ``resolved_params`` / ``results`` JSON blobs.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class JobState(str, Enum):
    """Lifecycle states. The string values are what gets persisted/returned."""

    PREPARED = "prepared"        # inputs written, not yet submitted
    SUBMITTED = "submitted"      # handed to scheduler, id assigned
    QUEUED = "queued"            # waiting in the scheduler queue
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }


# Fields stored as JSON-serialised columns rather than scalars.
JSON_FIELDS = ("resolved_params", "resources", "warnings", "results")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    """Short, collision-resistant id used both as PK and workdir name."""
    return f"dft-{uuid.uuid4().hex[:12]}"


@dataclass
class JobRecord:
    job_id: str
    engine: str
    calc_type: str
    state: str
    workdir: str
    label: Optional[str] = None
    scheduler_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    resolved_params: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict suitable for returning from an MCP tool."""
        return asdict(self)

    @classmethod
    def create(
        cls,
        engine: str,
        calc_type: str,
        workdir: str,
        label: Optional[str] = None,
        parent_job_id: Optional[str] = None,
    ) -> "JobRecord":
        return cls(
            job_id=new_job_id(),
            engine=engine,
            calc_type=calc_type,
            state=JobState.PREPARED.value,
            workdir=workdir,
            label=label,
            parent_job_id=parent_job_id,
        )
