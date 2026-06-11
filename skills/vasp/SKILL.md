---
name: vasp
description: >
  End-to-end guidance for running periodic plane-wave DFT with VASP on an HPC via the dft_* MCP tools — designing the calculation, submitting and monitoring jobs, triaging failures, and interpreting results back into a materials-discovery decision. Covers the full calculation range: relaxation, static/SCF, DOS, band structure, dielectric/optical, magnetism + Hubbard U, phonons, AIMD, and NEB. Use this skill WHENEVER the user wants to run, set up, debug, or interpret a VASP calculation, validate a candidate with DFT, compute a solid-state property (formation energy, band gap, elastic/phonon/DOS), verify an ML-predicted property flagged by candidate-screener, or asks anything about INCAR/KPOINTS/POTCAR/POSCAR, ENCUT/k-point convergence, SLURM submission of VASP, or electronic/ionic convergence problems — even if they don't explicitly say "VASP". For molecular quantum chemistry use the `orca` skill instead; for fast ML property prediction (not DFT) use the matgl/matcalc tools.
---

# VASP DFT Skill

This skill drives **periodic plane-wave DFT** calculations with VASP through the
`dft_*` MCP tools. Those tools run on an HPC that you cannot SSH into — they are
your only way to touch the cluster. Your job is the **judgment** the tools can't
encode: which calculation answers the question, which parameters are
scientifically appropriate, how to read a failure, and whether the result is
trustworthy enough to act on.

**Input:** a structure (POSCAR/CIF) — often a candidate flagged by the
`candidate-screener` skill for DFT verification.
**Output:** a converged, reliability-assessed property (formation energy,
stability, band gap, …) with full provenance, ready for `synthesis-planner`,
`stability-analyzer`, or another DFT step.

---

## Core principle: you are the judgment layer

The tools are the execution layer. They already know *how* to write a valid
INCAR (via pymatgen input sets), `sbatch` a job, poll the scheduler, and parse
`vasprun.xml`. They do **not** know *what* to run or whether it makes sense.
Keep that division in mind:

- **Don't** hand-write INCAR/KPOINTS/POTCAR or shell out to VASP yourself — the
  agent has no cluster access, and the tools encode tested defaults. Drive the
  tools; override their defaults only when the science demands it.
- **Do** own the decisions: calc type, functional, convergence, magnetism,
  resources, failure diagnosis, and the final scientific interpretation.

Because DFT jobs run for minutes to days, the tools are **non-blocking**: you
`prepare` → `submit` → poll `status` → `fetch_results`. A persistent `job_id`
ties these together across polling cycles and even across separate sessions, so
you never hold a blocking call open.

## The tools you drive

| Tool | Role in the lifecycle |
|------|----------------------|
| `dft_prepare_calculation` | Stage + validate inputs into a fresh workdir. **Consumes no HPC time** — this is your plan artifact. |
| `dft_submit_calculation` | Hand the prepared job to SLURM. The point of no (cheap) return. |
| `dft_get_calculation_status` | Poll job state (queued/running/completed/failed). Cheap, idempotent. |
| `dft_fetch_results` | Parse `vasprun.xml` into structured results once complete. |
| `dft_restart_calculation` | Clone a job, continuing from `CONTCAR`, for resubmission after a fix. |
| `dft_cancel_calculation` | Cancel a queued/running job. |

All take `engine="vasp"`. Supporting tools you'll often use first:
`matgl_relax_structure` (cheap ML pre-relaxation before expensive DFT),
`structure_validator` / `composition_analyzer` (sanity checks),
`mp_get_material_properties` (reference values), `ase_store_result` (cache).

---

## The lifecycle

Walk every calculation through these phases. They are the same for any
engine; the physics in phases 2, 5, and 6 is VASP-specific.

### 1. Intake
Establish **what question the DFT is answering** before touching parameters:
- Verifying an ML prediction from `candidate-screener`? Capture the target
  property, the ML value, and its confidence — you'll compare against it.
- A fresh property request? Identify the observable (formation energy,
  band gap, bulk modulus, magnetic ground state, phonon stability, …).
- Decide whether DFT is even warranted, or whether a cached MP value / ML
  estimate already suffices. DFT is expensive; spend it deliberately.

### 2. Design the calculation
Choose the calc type and parameters. This is the heart of the skill — read
[references/calculation-design.md](references/calculation-design.md) for the
full calc-type matrix and INCAR decision guidance. The essentials:

- **Pre-relax with ML first.** For any structure not already at its DFT
  equilibrium, run `matgl_relax_structure` before VASP. It removes bad contacts
  cheaply and cuts ionic steps dramatically.
- **Pick the calc type** (see the tiered matrix below).
- **Make the decisions the defaults can't:** smearing (`ISMEAR`/`SIGMA` —
  metal vs. insulator), spin (`ISPIN`/`MAGMOM`), Hubbard `LDAU`+U for
  correlated 3d-oxides/f-electrons, cell vs. ions (`ISIF`), functional, and
  pseudopotential choice. Express all of these through `overrides`.

### 3. Plan & confirm — **gate before spending HPC time**
After `dft_prepare_calculation` (which writes inputs but runs nothing), **stop
and present the plan to the user before `dft_submit_calculation`**:

> Show: calc type, the resolved INCAR/KPOINTS (from `resolved_params`), the
> structure/formula, the resource request, and a rough cost (node-hours, and
> for sweeps the **number of jobs**). Surface any `warnings` verbatim. Then ask
> for approval.

This matters because submission consumes shared-cluster time that can't be
un-spent, and a wrong `ENCUT` or k-mesh wastes hours. **Convergence sweeps and
batches especially** — propose the whole matrix and total cost, and only fan
out after a go-ahead. Never auto-submit expensive or many jobs.

### 4. Submit
On approval, `dft_submit_calculation(job_id, resources=...)`. Choose resources
deliberately: `nodes`, `ntasks`, `walltime`, `partition`. Match parallel INCAR
tags (`NCORE`/`KPAR`) to the node layout — see the design reference.

### 5. Monitor & triage
Poll `dft_get_calculation_status` (back off — minutes, not seconds; these are
long jobs). On a terminal `failed`, **diagnose before resubmitting** — read
[references/failure-triage.md](references/failure-triage.md). The big three:
electronic non-convergence (tune `ALGO`/mixing/`NELM`), ionic non-convergence
(restart from `CONTCAR`), and resource faults (OOM/walltime → `NCORE`/`KPAR`,
more resources). Apply the fix via `dft_restart_calculation` (it continues from
`CONTCAR` automatically) — and re-run the plan/confirm gate for the restart.

### 6. Interpret
`dft_fetch_results`, then judge — don't just forward numbers. Read
[references/result-interpretation.md](references/result-interpretation.md):
confirm `converged` (both electronic *and* ionic), check forces/stress are
small, and translate the raw output into the target observable. A non-converged
run is not a result.

### 7. Decide & hand back
Close the loop with the pipeline. If you were verifying an ML prediction,
state whether DFT **confirms or contradicts** it and upgrade the confidence
accordingly. Return the property with provenance (`job_id`, resolved params,
functional) so `synthesis-planner` / `stability-analyzer` / the discovery loop
can act. Cache via `ase_store_result` to avoid recomputation.

---

## Calculation types (the "full vision")

VASP supports far more than relaxation. Each type is tagged by **how it runs
today** so you never claim a calculation happened when it didn't — the
`dft_prepare_calculation` adapter will *warn loudly and fall back* if you pass
an unsupported `calc_type`, so trust that warning.

**Tier 1 — runnable via the `calc_type` template:**
`relax` · `static` · `single_point`

**Tier 2 — runnable now via `overrides` on a `static`/`relax` template**
(no new tooling; just INCAR/KPOINTS settings):
DOS & projected DOS · magnetism (`ISPIN`/`MAGMOM`) · Hubbard `LDAU`+U ·
dielectric/optical (`LOPTICS`) · Bader-charge prep · spin-orbit (`LSORBIT`).

**Tier 3 — design guidance only; NOT yet executable** through these tools
(would require extending `core/dft/engines/vasp.py`):
band structure (line-mode k-path) · phonons (finite-displacement / DFPT) ·
AIMD · NEB · elastic constants.

For Tier 3, advise the correct workflow but be explicit that it isn't yet
runnable here — don't submit a job that silently degrades to a relaxation.
The full recipe for each (including the exact `overrides`) is in
[references/calculation-design.md](references/calculation-design.md).

---

## Convergence testing — non-negotiable for production

DFT numbers are meaningless without it. Before a production calculation whose
absolute energy matters (formation energies, energy differences, EOS), converge
**ENCUT** and **k-point density** to the target tolerance. This is a small fan
of cheap jobs — exactly the case the plan/confirm gate is for: propose the sweep
matrix and total cost, get approval, submit as a batch, then read the trend.
Protocol and tolerances in
[references/convergence-testing.md](references/convergence-testing.md).

Skipping convergence testing is acceptable only for quick screening where you
explicitly flag the result as unconverged/qualitative.

---

## Overrides cheat-sheet

`overrides` is how you steer the calculation. Shape for VASP:

```json
{
  "encut": 520,
  "kpts": 64,                       // int = reciprocal density; or [4,4,4] explicit grid
  "incar": {"ISMEAR": 0, "SIGMA": 0.05, "ISPIN": 2, "LDAU": true},
  "vasp_pp_path": "/opt/vasp/potentials"
}
```

The tool starts from a pymatgen input set (`MPRelaxSet`/`MPStaticSet`) and
applies your `overrides` on top, so you only specify what should differ from
those sane defaults. Full tag-by-tag guidance lives in the design reference.

---

## Pipeline integration

- **Upstream:** `candidate-screener` flags "high-scoring ML predictions for DFT
  verification." Treat its output as your intake: structure + which property +
  the ML value/confidence to confirm.
- **Downstream:** hand a confirmed structure to `synthesis-planner`, feed
  energetics to `stability-analyzer`, or trigger the next DFT step. Always
  attach provenance.

## Reference files
- [references/calculation-design.md](references/calculation-design.md) — calc-type matrix, INCAR tag judgment, functionals, pseudopotentials, magnetism, +U, the overrides cookbook.
- [references/convergence-testing.md](references/convergence-testing.md) — ENCUT & k-point convergence protocol and tolerances.
- [references/failure-triage.md](references/failure-triage.md) — error → diagnosis → fix, and how to restart.
- [references/result-interpretation.md](references/result-interpretation.md) — reading results, reliability checks, mapping to observables.
