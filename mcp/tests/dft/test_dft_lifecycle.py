"""
End-to-end smoke tests for the DFT job-lifecycle tools.

These run without VASP or ORCA installed: the scheduler is set to ``local`` and
the engine "run command" is replaced with a shell snippet that fabricates a
plausible output file. This exercises the full prepare -> submit -> poll ->
fetch loop plus the persistence layer.
"""

import time

import pytest

from core.dft import reset_config_cache, reset_store_cache
from core.dft.config import get_config
from core.dft.models import JobRecord, JobState
from core.dft.store import get_job_store

H2_XYZ = """2

H   0.000000   0.000000   0.000000
H   0.000000   0.000000   0.740000
"""

SI_POSCAR = """Si
1.0
2.715 2.715 0.000
0.000 2.715 2.715
2.715 0.000 2.715
Si
2
direct
0.00 0.00 0.00
0.25 0.25 0.25
"""

# A fake ORCA run: emit an ORCA-looking output then exit 0.
_FAKE_ORCA_CMD = (
    "printf '"
    "FINAL SINGLE POINT ENERGY      -1.123456789\\n"
    "****ORCA TERMINATED NORMALLY****\\n"
    "' > orca.out"
)


@pytest.fixture()
def dft_env(tmp_path, monkeypatch):
    """Point the DFT subsystem at a temp workdir + sqlite DB, local scheduler."""
    workdir = tmp_path / "dft_jobs"
    db_url = f"sqlite:///{workdir/'jobs.db'}"
    monkeypatch.setenv("MATCLAW_DFT_WORKDIR", str(workdir))
    monkeypatch.setenv("MATCLAW_DFT_DATABASE_URL", db_url)
    monkeypatch.setenv("MATCLAW_DFT_SCHEDULER", "local")
    monkeypatch.delenv("MATCLAW_DFT_CONFIG", raising=False)
    reset_config_cache()
    reset_store_cache()
    # Swap ORCA's run command for the fake one (no binary required).
    cfg = get_config()
    cfg.engines.orca_command = _FAKE_ORCA_CMD
    yield cfg
    reset_config_cache()
    reset_store_cache()


def _poll_until_terminal(get_status, job_id, timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = get_status(job_id)
        if last.get("is_terminal"):
            return last
        time.sleep(0.2)
    return last


# -- persistence layer --------------------------------------------------------

def test_store_crud(dft_env):
    store = get_job_store()
    rec = JobRecord.create(engine="orca", calc_type="single_point", workdir="/tmp/x")
    store.create(rec)

    fetched = store.get(rec.job_id)
    assert fetched is not None
    assert fetched.engine == "orca"
    assert fetched.state == JobState.PREPARED.value

    store.update(rec.job_id, state=JobState.RUNNING.value, scheduler_id="999")
    assert store.get(rec.job_id).state == JobState.RUNNING.value
    assert store.get(rec.job_id).scheduler_id == "999"

    assert any(r.job_id == rec.job_id for r in store.list())
    assert store.delete(rec.job_id) is True
    assert store.get(rec.job_id) is None


def test_json_fields_roundtrip(dft_env):
    store = get_job_store()
    rec = JobRecord.create(engine="vasp", calc_type="relax", workdir="/tmp/y")
    rec.resolved_params = {"incar": {"ENCUT": 520}, "n_sites": 2}
    rec.warnings = ["a", "b"]
    store.create(rec)
    got = store.get(rec.job_id)
    assert got.resolved_params["incar"]["ENCUT"] == 520
    assert got.warnings == ["a", "b"]


# -- engine input preparation -------------------------------------------------

def test_prepare_orca(dft_env):
    from tools.dft import dft_prepare_calculation

    res = dft_prepare_calculation(engine="orca", structure=H2_XYZ, calc_type="single_point")
    assert res["success"] is True
    assert "orca.inp" in res["input_files"]
    inp = open(res["input_files"]["orca.inp"]).read()
    assert "* xyz 0 1" in inp
    assert "H" in inp


def test_prepare_vasp(dft_env):
    from tools.dft import dft_prepare_calculation

    res = dft_prepare_calculation(engine="vasp", structure=SI_POSCAR, calc_type="relax")
    assert res["success"] is True
    # POTCAR is typically unavailable in CI; INCAR/POSCAR must still be written.
    assert "INCAR" in res["input_files"]
    assert "POSCAR" in res["input_files"]
    assert res["resolved_params"]["formula"] == "Si"


def test_prepare_unknown_engine(dft_env):
    from tools.dft import dft_prepare_calculation

    res = dft_prepare_calculation(engine="gaussian", structure=H2_XYZ)
    assert res["success"] is False
    assert "Unknown engine" in res["error"]


def test_unsupported_calc_type_warns_not_silent(dft_env):
    """A Tier-3 calc_type must warn loudly, not silently masquerade as the default."""
    from tools.dft import dft_prepare_calculation

    res = dft_prepare_calculation(engine="orca", structure=H2_XYZ, calc_type="td-dft")
    assert res["success"] is True
    # Resolved calc_type reflects the ACTUAL fallback, not the request.
    assert res["resolved_params"]["calc_type"] == "single_point"
    assert any("not a runnable ORCA template" in w for w in res["warnings"])

    res2 = dft_prepare_calculation(engine="vasp", structure=SI_POSCAR, calc_type="phonon")
    assert res2["success"] is True
    assert res2["resolved_params"]["calc_type"] == "relax"
    assert any("not a runnable VASP template" in w for w in res2["warnings"])


# -- full lifecycle via the local scheduler -----------------------------------

def test_full_lifecycle_orca(dft_env):
    from tools.dft import (
        dft_fetch_results,
        dft_get_calculation_status,
        dft_prepare_calculation,
        dft_submit_calculation,
    )

    prep = dft_prepare_calculation(engine="orca", structure=H2_XYZ, calc_type="single_point")
    job_id = prep["job_id"]
    assert prep["state"] == JobState.PREPARED.value

    sub = dft_submit_calculation(job_id)
    assert sub["success"] is True
    assert sub["state"] == JobState.QUEUED.value
    assert sub["scheduler_id"]

    final = _poll_until_terminal(dft_get_calculation_status, job_id)
    assert final["state"] == JobState.COMPLETED.value

    out = dft_fetch_results(job_id)
    assert out["success"] is True
    assert out["ready"] is True
    assert out["results"]["parsed"] is True
    assert out["provenance"]["engine"] == "orca"


def test_fetch_before_ready(dft_env):
    from tools.dft import dft_fetch_results, dft_prepare_calculation

    prep = dft_prepare_calculation(engine="orca", structure=H2_XYZ, calc_type="single_point")
    out = dft_fetch_results(prep["job_id"])
    assert out["success"] is True
    assert out["ready"] is False  # prepared, not yet run


def test_restart_clones_job(dft_env):
    from tools.dft import dft_prepare_calculation, dft_restart_calculation

    prep = dft_prepare_calculation(engine="orca", structure=H2_XYZ, calc_type="single_point")
    rs = dft_restart_calculation(prep["job_id"])
    assert rs["success"] is True
    assert rs["parent_job_id"] == prep["job_id"]
    assert rs["job_id"] != prep["job_id"]
    assert rs["state"] == JobState.PREPARED.value
