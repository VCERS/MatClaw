# Interpreting VASP Results

`dft_fetch_results(job_id)` returns a structured dict for a completed VASP job.
Your job is to judge it, not just relay it — a number from a non-converged or
ill-posed run is worse than no number.

## What `dft_fetch_results` gives you
Key fields in `results` (parsed from `vasprun.xml`):
- `converged`, `converged_electronic`, `converged_ionic` (booleans)
- `final_energy_eV` (total energy of the cell)
- `n_ionic_steps`
- `formula`, `final_structure_cif`
- `output_files` present in the workdir

## Reliability checklist — run this before trusting anything
1. **Converged?** Require `converged_electronic` **and** (for a relax)
   `converged_ionic`. If either is false, it is not a result — triage and
   restart, don't report the energy.
2. **Geometry sane?** The `final_structure` should resemble the input (no
   exploded cell, no atoms merged). Compare volume/lattice to expectation.
3. **Forces/stress small?** For a relax, forces should be below your `EDIFFG`.
4. **Settings appropriate?** Was smearing right for the system (metal vs.
   insulator)? Was spin on if the system is magnetic? A converged run with the
   wrong `ISMEAR`/`ISPIN` gives a precise wrong answer.
5. **Convergence tested?** Is the energy from ENCUT/k-converged settings? If
   not, label it qualitative.

## Mapping results to observables

- **Total energy** — only comparable between calculations with the *same*
  functional, ENCUT, k-density, +U, and pseudopotentials. Never compare raw
  energies across inconsistent settings.
- **Formation energy** — `E(compound) − Σ µ(elements)` per formula unit, using
  consistent elemental references (or MP's correction scheme). This is usually
  the quantity `candidate-screener` wants verified.
- **Relative stability / decomposition** — compare formation energies on the
  convex hull; hand energetics to `stability-analyzer` rather than judging hull
  distance ad hoc.
- **Band gap** — from a static/DOS run. Remember semilocal DFT (PBE)
  *underestimates* gaps by ~30–50%; for quantitative gaps note this caveat or
  use HSE06. Distinguish the fundamental gap from the optical gap.
- **DOS / magnetic moment** — read from the DOS run and `MAGMOM` in OUTCAR;
  confirm the magnetic state matches the intended ordering.

## Closing the loop
Return to the pipeline with:
- the **observable** (not just total energy) and its units,
- a **reliability statement** (converged? convergence-tested? functional
  caveats?),
- the **verdict vs. the ML prediction** if this was a verification job —
  "DFT confirms the ML formation energy within X" or "DFT contradicts it;
  the candidate is/ isn't stable" — and an upgraded confidence,
- **provenance**: `job_id`, functional, ENCUT, k-density, +U, pseudopotentials.

Cache the structured result (`ase_store_result`) so the same calculation isn't
repeated, and state the recommended next step (synthesis planning, further DFT,
or rejection).
