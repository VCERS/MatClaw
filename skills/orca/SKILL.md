---
name: orca
description: >
  End-to-end guidance for running molecular quantum-chemistry with ORCA on an HPC via the dft_* MCP tools, plus ORCA-specific post-processing (orca_* tools) — designing the calculation (method/basis/solvation), submitting and monitoring jobs, triaging SCF/geometry failures, interpreting outputs, and generating visualization cubes. Covers single-point energies, geometry optimization, frequencies/thermochemistry, and design guidance for TD-DFT, scans, NEB-TS, and multireference. Use this skill WHENEVER the user wants to run, set up, debug, or interpret an ORCA calculation on a molecule; compute a molecular property (energy, HOMO/LUMO gap, dipole, frequencies, reaction/binding energy); optimize or verify a molecular geometry; analyze an existing ORCA output directory (.out/.gbw); generate HOMO/LUMO, electron-density, or ESP cube files; or asks about ORCA input (! keywords, %blocks), basis sets (def2-*), functionals, charge/multiplicity, SCF convergence, or orca_plot — even if they don't say "ORCA". For periodic/crystalline solids use the `vasp` skill instead.
---

# ORCA Quantum Chemistry Skill

This skill drives **molecular** quantum-chemistry calculations with ORCA through
the `dft_*` MCP tools, and handles ORCA-specific **post-processing** (output
analysis and cube generation) through the `orca_*` tools. Everything runs on an
HPC you cannot SSH into — the tools are your only access. Your job is the
**judgment** the tools don't encode: which method/basis fits the question, the
right charge and spin state, how to read an SCF failure, whether a geometry is a
true minimum, and what the numbers mean.

**Input:** a molecular structure (XYZ), or an existing ORCA calculation
directory to analyze. **Output:** a converged, reliability-assessed molecular
property and/or visualization artifacts, with provenance.

---

## Core principle: you are the judgment layer

The tools are the execution layer — they write a valid `orca.inp`, `sbatch` the
job, poll, parse the output, and run `orca_plot`. They do **not** decide the
chemistry. So:

- **Don't** hand-write input files or shell out to ORCA/`orca_plot` yourself —
  no cluster access, and the tools are tested. Drive the tools.
- **Do** own: method/basis/solvation, charge & multiplicity, whether a
  frequency job is needed to confirm a minimum, SCF-failure diagnosis, and the
  scientific interpretation.

ORCA jobs are long, so the run tools are **non-blocking**: `prepare` → `submit`
→ poll `status` → `fetch_results`, tied together by a persistent `job_id`.

## Two tool families

**Running calculations** (engine-agnostic lifecycle, `engine="orca"`):

| Tool | Role |
|------|------|
| `dft_prepare_calculation` | Stage + validate `orca.inp`. **No HPC time** — your plan artifact. |
| `dft_submit_calculation` | Submit to SLURM. |
| `dft_get_calculation_status` | Poll job state. |
| `dft_fetch_results` | Parse `orca.out` (delegates to `orca_summarize_output`). |
| `dft_restart_calculation` | Clone a job, copying `.gbw` for an initial-guess restart. |
| `dft_cancel_calculation` | Cancel a queued/running job. |

**ORCA-specific post-processing** (analysis + visualization on `.out`/`.gbw`):

| Tool | Role |
|------|------|
| `orca_summarize_output` / `orca_batch_summarize_outputs` | Structured summary of one / many outputs. |
| `orca_scan_output_files` / `orca_pick_output` | Discover / select the right `.out` in a directory. |
| `orca_validate_environment` / `orca_validate_calc_dir` | Preflight gate for cube generation. |
| `orca_find_matching_gbw` | Match a `.gbw` to a chosen output. |
| `orca_generate_homo_lumo_cubes` / `orca_generate_density_esp_cubes` / `orca_generate_mo_cube` | Generate visualization cubes via `orca_plot`. |

---

## The lifecycle

### 1. Intake
Decide which of two modes you're in:
- **Run a new calculation** — you have a molecule and a question.
- **Analyze / visualize an existing calculation** — the user points at a
  directory of `.out`/`.gbw` files (possibly produced elsewhere). Jump to
  [Output analysis](references/output-analysis.md) or
  [Cube generation](references/cube-generation.md); these don't need the run
  lifecycle.

For a new run, capture the question (energy? optimized geometry? is it a
minimum? frontier-orbital picture? reaction/binding energy?) and any handoff
from `candidate-screener`.

### 2. Design the calculation
The chemistry decisions — read
[references/calculation-design.md](references/calculation-design.md):
- **Charge & multiplicity** — get these right first; everything else is wrong if
  they're wrong. Multiplicity = 2S+1 (1 = closed-shell singlet, 2 = doublet
  radical, 3 = triplet…).
- **Method** — functional family (e.g. `B3LYP`, `PBE0`, `r²SCAN-3c`,
  `wB97X-V`), almost always with a **dispersion** correction (`D3BJ`/`D4`).
- **Basis set** — `def2-SVP` (fast/screening) → `def2-TZVP` (production) →
  `def2-QZVP` (benchmark); add diffuse (`def2-TZVPD`) for anions/excited states.
- **Acceleration & accuracy** — `RIJCOSX` + auxiliary basis for hybrids,
  `TightSCF`, `Grid` settings.
- **Solvation** — implicit `CPCM(solvent)` or `SMD(solvent)` for condensed phase.

All of these are expressed through `overrides` (`method`, `basis`, `keywords`,
`blocks`) — see the cheat-sheet below.

### 3. Plan & confirm — **gate before spending HPC time**
After `dft_prepare_calculation` (writes `orca.inp`, runs nothing), **stop and
show the user the plan before `dft_submit_calculation`**: the resolved simple-
input line (`! method basis keywords`), charge/multiplicity, calc type, the
geometry, resources, and a rough cost. Surface `warnings` verbatim (open-shell,
unsupported calc_type fallback, periodic-structure-into-ORCA, …). Get approval.
This is essential for expensive jobs (large basis, `Freq`, many conformers) and
for any batch — propose the whole set and total cost before fanning out.

### 4. Submit
On approval, `dft_submit_calculation(job_id, resources=...)`. For parallel runs
set `overrides.nprocs` (writes `%pal nprocs N end`) and a matching `ntasks`.

### 5. Monitor & triage
Poll `dft_get_calculation_status` (back off — minutes). On failure, diagnose
before resubmitting — read [references/failure-triage.md](references/failure-triage.md):
SCF non-convergence (`SlowConv`, `KDIIS`, better guess), geometry trouble
(restart from `.gbw`/last geometry), open-shell/spin-contamination. Apply fixes
via `dft_restart_calculation` and re-run the plan/confirm gate.

### 6. Interpret
`dft_fetch_results` returns a structured summary (normal termination, final
energy, HOMO/LUMO + gap, imaginary-frequency count, reliability warnings).
Judge it — read [references/output-analysis.md](references/output-analysis.md).
Two molecular-specific musts:
- **Verify minima with frequencies.** A bare `Opt` does not prove a minimum.
  Run `freq` (or `opt_freq`) and require **zero imaginary frequencies** before
  trusting an optimized geometry; one imaginary mode = a transition state.
- **Heed open-shell warnings.** For radicals/UHF, frontier-orbital and
  spin-contamination caveats matter; don't over-state HOMO/LUMO assignments.

### 7. Visualize (optional) & hand back
If the user wants orbital/density/ESP pictures, run the **guarded cube workflow**
— read [references/cube-generation.md](references/cube-generation.md). Then close
the loop with the pipeline: report the property with provenance, the verdict vs.
any ML prediction, and the recommended next step.

---

## Calculation types (the "full vision")

Tagged by how each runs today. An unsupported `calc_type` passed to
`dft_prepare_calculation` **warns loudly and falls back to `single_point`** —
trust that warning, don't assume the intended job ran.

**Tier 1 — runnable via the `calc_type` template:**
`single_point` · `opt` · `freq` · `opt_freq`

**Tier 2 — runnable now via `overrides` (keywords/`%`-blocks)** on a Tier-1
template, no new tooling:
implicit solvation `CPCM`/`SMD` · dispersion `D3BJ`/`D4` · `RIJCOSX` ·
`TightSCF`/grid control · larger/diffuse basis sets · relaxed/rigid scans
(`%geom Scan ... end` on an `opt`) · constrained optimization.

**Tier 3 — design guidance only; NOT yet executable** (would need to extend
`core/dft/engines/orca.py`):
TD-DFT / excited states · NEB-TS transition-state search · CASSCF/NEVPT2
multireference · property jobs needing dedicated blocks (NMR, polarizability).

For Tier 3, advise the correct ORCA setup but state it can't be submitted
through the current tools yet. Recipes and exact keywords in
[references/calculation-design.md](references/calculation-design.md).

---

## Overrides cheat-sheet

```json
{
  "method": "B3LYP",
  "basis": "def2-TZVP",
  "keywords": "D3BJ RIJCOSX TightSCF",     // appended to the ! line
  "blocks": ["%cpcm smd true SMDsolvent \"water\" end"],  // arbitrary %-blocks
  "nprocs": 16
}
```

The tool builds the `! ...` simple-input line from `method` + `basis` +
`keywords` + the calc-type keyword (`Opt`/`Freq`/…), then appends your
`blocks` and the `* xyz charge mult` geometry. You specify only what differs
from the `B3LYP/def2-SVP` default. Always echo the resolved input line at the
plan/confirm gate.

---

## Post-processing: analysis & cube generation

Two ORCA-only capabilities the run lifecycle doesn't cover, each with its own
reference:

- **Output analysis** — summarize a single output or batch-summarize a tree of
  results; interpret convergence/energy/orbitals/frequencies with reliability
  warnings. See [references/output-analysis.md](references/output-analysis.md).
- **Cube generation** — a **guarded** workflow (preflight environment + directory
  checks before invoking `orca_plot`) for HOMO/LUMO, electron-density, and ESP
  cubes. See [references/cube-generation.md](references/cube-generation.md).
  Always run the triage gate (`orca_validate_environment` →
  `orca_validate_calc_dir`) first and stop if the directory isn't cube-ready.

These work on *any* ORCA output directory, including calculations not produced
through the run lifecycle — handy when the user just hands you results.

## Pipeline integration
- **Upstream:** molecular candidates / properties flagged by `candidate-screener`
  for verification (e.g. a binding or redox property). Capture the target and
  ML value to compare against.
- **Downstream:** return the confirmed property + provenance; hand off to the
  next discovery step. Cache via `ase_store_result`.

## Reference files
- [references/calculation-design.md](references/calculation-design.md) — method/basis/solvation/dispersion, charge & multiplicity, calc-type matrix, overrides cookbook.
- [references/output-analysis.md](references/output-analysis.md) — summarizing & interpreting outputs, reliability warnings, frequency/minimum checks.
- [references/cube-generation.md](references/cube-generation.md) — the guarded HOMO/LUMO & density/ESP cube workflow.
- [references/failure-triage.md](references/failure-triage.md) — SCF/geometry/open-shell failures and restarts.
