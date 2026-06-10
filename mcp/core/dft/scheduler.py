"""
Scheduler abstraction — the engine-agnostic control plane.

Every method here is fast and non-blocking; the long-running computation lives
in the batch system, decoupled from the agent's session. Two backends are
provided:

* :class:`SlurmScheduler` — production HPC use (``sbatch`` / ``squeue`` /
  ``sacct`` / ``scancel``).
* :class:`LocalScheduler` — runs the batch script as a detached local process.
  No queue required, so the whole submit -> poll -> fetch loop is testable on a
  laptop or CI runner.

The backend is selected by ``scheduler.type`` in the configuration.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from .config import DFTConfig, get_config
from .models import JobRecord, JobState

# How scheduler-reported states map onto our lifecycle states.
_SLURM_STATE_MAP = {
    "PENDING": JobState.QUEUED,
    "CONFIGURING": JobState.QUEUED,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "TIMEOUT": JobState.FAILED,
    "BOOT_FAIL": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
    "DEADLINE": JobState.CANCELLED,
    "PREEMPTED": JobState.FAILED,
}

EXIT_CODE_FILE = ".dft_exit_code"
PID_FILE = ".dft_pid"
RUN_LOG = "run.log"


class Scheduler(ABC):
    """Control-plane interface. Implementations must stay non-blocking."""

    def __init__(self, config: DFTConfig):
        self.config = config

    @abstractmethod
    def build_submit_script(
        self, record: JobRecord, run_commands: List[str], resources: Dict
    ) -> str:
        """Return the full batch-script text for this job."""

    @abstractmethod
    def submit(self, workdir: str, script_path: str) -> str:
        """Submit the script; return the scheduler's job id."""

    @abstractmethod
    def status(self, scheduler_id: str, workdir: str) -> JobState:
        """Return the current lifecycle state for a submitted job."""

    @abstractmethod
    def cancel(self, scheduler_id: str, workdir: str) -> bool:
        """Request cancellation; return True if the request was accepted."""


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


class SlurmScheduler(Scheduler):
    """SLURM backend via the standard CLI."""

    def build_submit_script(
        self, record: JobRecord, run_commands: List[str], resources: Dict
    ) -> str:
        sc = self.config.scheduler
        job_name = record.label or record.job_id
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={record.workdir}/slurm-%j.out",
            f"#SBATCH --error={record.workdir}/slurm-%j.err",
            f"#SBATCH --nodes={resources.get('nodes', sc.default_nodes)}",
            f"#SBATCH --ntasks={resources.get('ntasks', sc.default_ntasks)}",
            f"#SBATCH --time={resources.get('walltime', sc.default_walltime)}",
        ]
        partition = resources.get("partition", sc.partition)
        account = resources.get("account", sc.account)
        qos = resources.get("qos", sc.qos)
        if partition:
            directives.append(f"#SBATCH --partition={partition}")
        if account:
            directives.append(f"#SBATCH --account={account}")
        if qos:
            directives.append(f"#SBATCH --qos={qos}")

        body = [
            "",
            "set -e",
            *[f"module load {m}" for m in sc.modules],
            *sc.prologue,
            f'cd "{record.workdir}"',
            "",
            *run_commands,
        ]
        return "\n".join(directives + body) + "\n"

    def submit(self, workdir: str, script_path: str) -> str:
        result = _run([self.config.scheduler.submit_command, script_path], cwd=workdir)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
        # "Submitted batch job 1234567"
        match = re.search(r"(\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job id from sbatch output: {result.stdout!r}")
        return match.group(1)

    def status(self, scheduler_id: str, workdir: str) -> JobState:
        sc = self.config.scheduler
        # squeue knows about pending/running jobs.
        squeue = _run([sc.status_command, "-j", scheduler_id, "-h", "-o", "%T"])
        token = squeue.stdout.strip().splitlines()
        if token and token[0].strip():
            return _SLURM_STATE_MAP.get(token[0].strip().upper(), JobState.UNKNOWN)
        # Finished jobs leave squeue; consult the accounting database.
        sacct = _run(
            [sc.accounting_command, "-j", scheduler_id, "-n", "-P", "-o", "State"]
        )
        for line in sacct.stdout.splitlines():
            state = line.split("|")[0].strip().upper().split()[0] if line.strip() else ""
            if state:
                return _SLURM_STATE_MAP.get(state, JobState.UNKNOWN)
        return JobState.UNKNOWN

    def cancel(self, scheduler_id: str, workdir: str) -> bool:
        result = _run([self.config.scheduler.cancel_command, scheduler_id])
        return result.returncode == 0


class LocalScheduler(Scheduler):
    """Runs the batch script as a detached background process (no queue).

    Useful for development, CI, and small molecular ORCA jobs on a workstation.
    Status is derived from the process pid and an exit-code sentinel file.
    """

    def build_submit_script(
        self, record: JobRecord, run_commands: List[str], resources: Dict
    ) -> str:
        sc = self.config.scheduler
        lines = [
            "#!/bin/bash",
            *[f"module load {m}" for m in sc.modules],
            *sc.prologue,
            f'cd "{record.workdir}"',
            "{",
            *run_commands,
            f'}} > "{record.workdir}/{RUN_LOG}" 2>&1',
            f'echo $? > "{record.workdir}/{EXIT_CODE_FILE}"',
        ]
        return "\n".join(lines) + "\n"

    def submit(self, workdir: str, script_path: str) -> str:
        # Clear any stale sentinel from a previous run in this dir.
        exit_file = Path(workdir) / EXIT_CODE_FILE
        if exit_file.exists():
            exit_file.unlink()
        proc = subprocess.Popen(
            ["bash", script_path],
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from the MCP server process group
        )
        (Path(workdir) / PID_FILE).write_text(str(proc.pid))
        return str(proc.pid)

    def status(self, scheduler_id: str, workdir: str) -> JobState:
        exit_file = Path(workdir) / EXIT_CODE_FILE
        if exit_file.exists():
            code = exit_file.read_text().strip()
            return JobState.COMPLETED if code == "0" else JobState.FAILED
        # No sentinel yet — is the process still alive?
        try:
            pid = int(scheduler_id)
        except ValueError:
            return JobState.UNKNOWN
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # Gone without writing an exit code (killed/crashed).
            return JobState.UNKNOWN
        except PermissionError:
            return JobState.RUNNING
        return JobState.RUNNING

    def cancel(self, scheduler_id: str, workdir: str) -> bool:
        try:
            pid = int(scheduler_id)
        except ValueError:
            return False
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            return True  # already gone
        except OSError:
            return False
        return True


_BACKENDS = {"slurm": SlurmScheduler, "local": LocalScheduler}


def get_scheduler(config: Optional[DFTConfig] = None) -> Scheduler:
    cfg = config or get_config()
    backend = _BACKENDS.get(cfg.scheduler.type.lower())
    if backend is None:
        raise ValueError(
            f"Unknown scheduler type {cfg.scheduler.type!r}; "
            f"expected one of {sorted(_BACKENDS)}"
        )
    return backend(cfg)
