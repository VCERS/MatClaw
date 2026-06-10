"""
DFT job-lifecycle MCP tools (VASP + ORCA via engine dispatch).

These six tools are the agent-facing control plane for running DFT/QC jobs on an
HPC the agent cannot SSH into. Each call is short and non-blocking; the long
calculation runs under the batch scheduler, and a persistent ``job_id`` ties the
short calls together across polling cycles and even separate agent sessions.

    prepare_calculation     stage validated inputs into a fresh working dir
    submit_calculation      hand the prepared job to the scheduler
    get_calculation_status  poll the scheduler and update the job record
    fetch_results           parse completed outputs into structured results
    cancel_calculation      cancel a queued/running job
    restart_calculation     clone a job (with checkpoint) for resubmission

The physics differs by engine; the lifecycle does not. ``engine`` is a
discriminator ("vasp" | "orca"), not a separate tool family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field

# Subsystem (infrastructure) lives under core.dft; these tools only orchestrate.
from core.dft import get_config, get_engine, get_job_store, get_scheduler, supported_engines
from core.dft.models import JobRecord, JobState, _now


def _err(message: str, **extra) -> Dict[str, Any]:
    return {"success": False, "error": message, **extra}


def _refresh_status(record: JobRecord) -> JobRecord:
    """Poll the scheduler for a non-terminal job and persist any state change."""
    store = get_job_store()
    state = JobState(record.state) if record.state in JobState._value2member_map_ else JobState.UNKNOWN
    if not record.scheduler_id or state.is_terminal:
        return record
    scheduler = get_scheduler()
    try:
        new_state = scheduler.status(record.scheduler_id, record.workdir)
    except Exception as exc:
        return store.update(record.job_id, message=f"Status query failed: {exc}") or record
    if new_state == JobState.UNKNOWN:
        return record
    patch: Dict[str, Any] = {"state": new_state.value}
    if new_state.is_terminal:
        patch["completed_at"] = _now()
    return store.update(record.job_id, **patch) or record


def prepare_calculation(
    engine: Annotated[str, Field(description='DFT engine: "vasp" (periodic) or "orca" (molecular).')],
    structure: Annotated[str, Field(description="Structure as CIF/POSCAR (VASP) or XYZ (ORCA) text.")],
    calc_type: Annotated[str, Field(description='Calculation type. VASP: relax|static|single_point. ORCA: single_point|opt|freq|opt_freq.')] = "relax",
    structure_format: Annotated[str, Field(description='Structure format: "auto" (default), "cif", "poscar", or "xyz".')] = "auto",
    charge: Annotated[int, Field(description="Total charge (ORCA molecular calcs).")] = 0,
    multiplicity: Annotated[int, Field(description="Spin multiplicity 2S+1 (ORCA molecular calcs).")] = 1,
    overrides: Annotated[Optional[Dict[str, Any]], Field(description='Engine-specific overrides, e.g. VASP {"encut":520,"kpts":64,"incar":{"ISMEAR":0}} or ORCA {"method":"B3LYP","basis":"def2-TZVP","keywords":"D3BJ"}.')] = None,
    label: Annotated[Optional[str], Field(description="Human-friendly label for the job.")] = None,
) -> Dict[str, Any]:
    """Stage and validate inputs for a DFT job without submitting it.

    Creates a fresh working directory, writes the engine input files, records a
    job in the persistent store with state ``prepared``, and returns the
    ``job_id`` to pass to ``submit_calculation``. No HPC resources are consumed.

    Returns a dict with: success, job_id, state, engine, calc_type, workdir,
    input_files, resolved_params, warnings.
    """
    try:
        if engine.lower() not in supported_engines():
            return _err(f"Unknown engine {engine!r}; expected one of {supported_engines()}")
        cfg = get_config()
        eng = get_engine(engine)
        store = get_job_store()

        record = JobRecord.create(
            engine=engine.lower(), calc_type=calc_type,
            workdir="",  # set below once we know the job id
            label=label,
        )
        record.workdir = str(cfg.resolved_workdir() / record.job_id)
        Path(record.workdir).mkdir(parents=True, exist_ok=True)

        prep = eng.prepare(
            workdir=record.workdir,
            structure=structure,
            calc_type=calc_type,
            structure_format=structure_format,
            charge=charge,
            multiplicity=multiplicity,
            overrides=overrides,
        )
        record.resolved_params = prep.resolved_params
        record.warnings = prep.warnings
        store.create(record)

        return {
            "success": True,
            "job_id": record.job_id,
            "state": record.state,
            "engine": record.engine,
            "calc_type": prep.resolved_params.get("calc_type", calc_type),
            "workdir": record.workdir,
            "input_files": prep.input_files,
            "resolved_params": prep.resolved_params,
            "warnings": prep.warnings,
            "next_step": "Call submit_calculation(job_id) to queue the job.",
        }
    except Exception as exc:
        return _err(f"prepare_calculation failed: {exc}")


def submit_calculation(
    job_id: Annotated[str, Field(description="Job id returned by prepare_calculation.")],
    resources: Annotated[Optional[Dict[str, Any]], Field(description='Scheduler overrides, e.g. {"nodes":1,"ntasks":32,"walltime":"12:00:00","partition":"gpu"}. Defaults come from config.')] = None,
) -> Dict[str, Any]:
    """Submit a prepared job to the scheduler and return its scheduler id.

    Builds the batch script from the engine's run command(s) plus the resolved
    resource request, submits it, and moves the job to ``queued``. Non-blocking.

    Returns a dict with: success, job_id, scheduler_id, state, submit_script.
    """
    try:
        cfg = get_config()
        store = get_job_store()
        record = store.get(job_id)
        if record is None:
            return _err(f"No such job: {job_id}")
        if record.state not in (JobState.PREPARED.value, JobState.FAILED.value):
            return _err(
                f"Job {job_id} is in state {record.state!r}; only 'prepared' or "
                "'failed' jobs can be submitted. Use restart_calculation to clone it."
            )

        resources = {**(record.resources or {}), **(resources or {})}
        eng = get_engine(record.engine)
        scheduler = get_scheduler()

        run_commands = eng.run_commands(record.workdir, record.resolved_params, resources, cfg)
        script_text = scheduler.build_submit_script(record, run_commands, resources)
        script_path = str(Path(record.workdir) / "submit.sh")
        Path(script_path).write_text(script_text)

        scheduler_id = scheduler.submit(record.workdir, script_path)
        store.update(
            job_id,
            state=JobState.QUEUED.value,
            scheduler_id=scheduler_id,
            resources=resources,
            submitted_at=_now(),
            message=None,
        )
        return {
            "success": True,
            "job_id": job_id,
            "scheduler_id": scheduler_id,
            "state": JobState.QUEUED.value,
            "submit_script": script_path,
            "next_step": "Poll with get_calculation_status(job_id).",
        }
    except Exception as exc:
        return _err(f"submit_calculation failed: {exc}")


def get_calculation_status(
    job_id: Annotated[str, Field(description="Job id to poll.")],
) -> Dict[str, Any]:
    """Poll the scheduler for a job and return its current lifecycle state.

    Cheap and idempotent — safe to call on a polling loop. Updates the persisted
    record when the scheduler reports a state change.

    Returns a dict with: success, job_id, state, scheduler_id, engine, calc_type,
    is_terminal, workdir, updated_at, message.
    """
    try:
        store = get_job_store()
        record = store.get(job_id)
        if record is None:
            return _err(f"No such job: {job_id}")
        record = _refresh_status(record)
        state = JobState(record.state) if record.state in JobState._value2member_map_ else JobState.UNKNOWN
        return {
            "success": True,
            "job_id": job_id,
            "state": record.state,
            "is_terminal": state.is_terminal,
            "scheduler_id": record.scheduler_id,
            "engine": record.engine,
            "calc_type": record.calc_type,
            "workdir": record.workdir,
            "updated_at": record.updated_at,
            "message": record.message,
            "next_step": (
                "Call fetch_results(job_id)." if state == JobState.COMPLETED
                else "Job finished without success; inspect logs or restart_calculation." if state.is_terminal
                else "Keep polling get_calculation_status(job_id)."
            ),
        }
    except Exception as exc:
        return _err(f"get_calculation_status failed: {exc}")


def fetch_results(
    job_id: Annotated[str, Field(description="Job id whose results to parse.")],
) -> Dict[str, Any]:
    """Parse a completed job's outputs into structured results.

    Refreshes status first. If the job is not yet complete, returns success=True
    with ``ready=False`` and the current state rather than an error. For ORCA
    this delegates to the existing tools.orca summariser; for VASP it parses
    vasprun.xml.

    Returns a dict with: success, ready, job_id, state, engine, calc_type,
    results, artifacts, provenance, warnings.
    """
    try:
        store = get_job_store()
        record = store.get(job_id)
        if record is None:
            return _err(f"No such job: {job_id}")
        record = _refresh_status(record)
        state = JobState(record.state) if record.state in JobState._value2member_map_ else JobState.UNKNOWN

        if state != JobState.COMPLETED:
            return {
                "success": True,
                "ready": False,
                "job_id": job_id,
                "state": record.state,
                "message": f"Results not available; job is {record.state!r}.",
            }

        eng = get_engine(record.engine)
        results = eng.parse_results(record.workdir, record.calc_type, record.resolved_params)
        store.update(job_id, results=results)

        return {
            "success": True,
            "ready": True,
            "job_id": job_id,
            "state": record.state,
            "engine": record.engine,
            "calc_type": record.calc_type,
            "results": results,
            "artifacts": {"workdir": record.workdir},
            "provenance": {
                "engine": record.engine,
                "calc_type": record.calc_type,
                "resolved_params": record.resolved_params,
                "scheduler_id": record.scheduler_id,
                "submitted_at": record.submitted_at,
                "completed_at": record.completed_at,
                "parent_job_id": record.parent_job_id,
            },
            "warnings": record.warnings,
        }
    except Exception as exc:
        return _err(f"fetch_results failed: {exc}")


def cancel_calculation(
    job_id: Annotated[str, Field(description="Job id to cancel.")],
) -> Dict[str, Any]:
    """Cancel a queued or running job via the scheduler.

    Returns a dict with: success, job_id, state, cancelled.
    """
    try:
        store = get_job_store()
        record = store.get(job_id)
        if record is None:
            return _err(f"No such job: {job_id}")
        if not record.scheduler_id:
            return _err(f"Job {job_id} has no scheduler id (not submitted).")
        scheduler = get_scheduler()
        ok = scheduler.cancel(record.scheduler_id, record.workdir)
        if ok:
            store.update(job_id, state=JobState.CANCELLED.value, completed_at=_now())
        return {
            "success": ok,
            "job_id": job_id,
            "state": JobState.CANCELLED.value if ok else record.state,
            "cancelled": ok,
        }
    except Exception as exc:
        return _err(f"cancel_calculation failed: {exc}")


def restart_calculation(
    job_id: Annotated[str, Field(description="Job id to restart from (typically failed or completed).")],
    overrides: Annotated[Optional[Dict[str, Any]], Field(description="Optional engine-specific overrides for the restarted job.")] = None,
    label: Annotated[Optional[str], Field(description="Label for the new job; defaults to the parent's label + '-restart'.")] = None,
) -> Dict[str, Any]:
    """Clone a job into a new prepared job, wiring in any checkpoint.

    VASP continues from CONTCAR; ORCA copies the .gbw for an initial guess. The
    new job starts in state ``prepared`` — submit it with submit_calculation.

    Returns a dict with: success, job_id (new), parent_job_id, state, workdir,
    input_files, warnings.
    """
    try:
        cfg = get_config()
        store = get_job_store()
        parent = store.get(job_id)
        if parent is None:
            return _err(f"No such job: {job_id}")

        eng = get_engine(parent.engine)
        child = JobRecord.create(
            engine=parent.engine,
            calc_type=parent.calc_type,
            workdir="",
            label=label or (f"{parent.label}-restart" if parent.label else None),
            parent_job_id=parent.job_id,
        )
        child.workdir = str(cfg.resolved_workdir() / child.job_id)

        prep = eng.prepare_restart(parent.workdir, child.workdir, parent.resolved_params, overrides)
        child.resolved_params = prep.resolved_params
        child.warnings = prep.warnings
        store.create(child)

        return {
            "success": True,
            "job_id": child.job_id,
            "parent_job_id": parent.job_id,
            "state": child.state,
            "workdir": child.workdir,
            "input_files": prep.input_files,
            "warnings": prep.warnings,
            "next_step": "Submit the restarted job with submit_calculation(job_id).",
        }
    except Exception as exc:
        return _err(f"restart_calculation failed: {exc}")
