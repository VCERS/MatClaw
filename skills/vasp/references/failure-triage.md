# VASP Failure Triage

When `dft_get_calculation_status` returns `failed` (or a job "completes" but
didn't converge), diagnose the cause before resubmitting. Blind restarts waste
HPC time and can mask a real problem. Inspect the job's `workdir` artifacts
(OUTCAR, OSZICAR, vasprun.xml, the SLURM `.err`) via the returned paths, match
the symptom below, apply the fix through `dft_restart_calculation` (which
continues from `CONTCAR`), and re-run the plan/confirm gate.

## The three big failure classes

### 1. Electronic (SCF) non-convergence
**Symptom:** `NELM` electronic steps reached without `EDIFF`; OSZICAR shows the
energy oscillating or crawling.
**Fixes (in order):**
- Change the mixer/algorithm: `ALGO=Normal` → `All` or `Damped` for metals/
  magnets; `AMIX=0.2, BMIX=0.0001` for hard cases.
- Raise `NELM` (e.g. 150) — but only if it's *approaching* convergence.
- Improve the start: better `MAGMOM`, a sensible `ISMEAR`/`SIGMA` for the
  system type, or read a converged `WAVECAR`/`CHGCAR` from a coarser run.
- For metals, a too-small `SIGMA` or tetrahedron smearing during relaxation
  causes this — switch to `ISMEAR=1, SIGMA=0.2`.

### 2. Ionic non-convergence
**Symptom:** `NSW` ionic steps exhausted; forces still above `EDIFFG`; geometry
drifting or oscillating.
**Fixes:**
- Restart from `CONTCAR` (`dft_restart_calculation` does this) — continuing from
  the last geometry usually converges.
- If forces oscillate, switch optimizer: `IBRION=1` (RMM-DIIS, near a minimum)
  vs. `IBRION=2` (CG, far from it); reduce `POTIM`.
- Pre-relax with `matgl_relax_structure` first if the input geometry was poor.
- Check for a too-loose `EDIFF` (electronic noise prevents ionic convergence) —
  tighten to 1e-6.

### 3. Resource / runtime faults
**Symptom:** segfault, OOM-killer, or SLURM `TIMEOUT` in the `.err`/status.
**Fixes:**
- Out of memory: reduce `NCORE`, increase nodes/memory, or set `LREAL=Auto` for
  large cells.
- Walltime exceeded: request more `walltime`, or restart from `CONTCAR` to
  continue a long relaxation across jobs.
- Crash on a specific node: often `KPAR`/`NCORE` not dividing the rank count —
  fix the factorization to match `ntasks`.

## Other common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `POTCAR not found` / POTCAR warning | pseudopotential path unset | set `vasp_pp_path` in config/overrides |
| `ZBRENT: fatal error` | bad geometry / too-large step | restart from CONTCAR, smaller `POTIM` |
| `Sub-Space-Matrix is not hermitian` | numerical/`ALGO` issue | `ALGO=Normal`, lower `AMIX` |
| Huge final energy / absurd structure | disordered or broken input | re-check structure (`structure_validator`); order it first |
| Converged but wrong magnetic moment | bad initial `MAGMOM`/no `ISPIN` | set `ISPIN=2` + proper `MAGMOM`, retry |

## Restart mechanics
`dft_restart_calculation(job_id, overrides=...)` clones the job into a fresh
workdir, copies `CONTCAR → POSCAR` (continuing the geometry) plus
INCAR/KPOINTS/POTCAR, and returns a new `job_id` in state `prepared`. Apply your
fix as `overrides`, then go through plan/confirm → submit again. Always tell the
user *why* it failed and *what* you changed — a silent restart hides the
diagnosis.
