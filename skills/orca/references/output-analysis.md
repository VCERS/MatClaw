# ORCA Output Analysis

Interpreting ORCA results — both for jobs you ran (`dft_fetch_results` delegates
here) and for existing `.out` files a user hands you. The parsing is done by
tested tools; your job is to judge reliability and translate to chemistry.

## Choosing the right tools

- **Single known `.out` file** → `orca_summarize_output(out_file)`.
- **A directory, unsure which output** → `orca_scan_output_files(root_dir)` to
  discover, then `orca_pick_output(calc_dir, preference="optimization")` to
  select, then summarize.
- **A whole tree of results** → `orca_batch_summarize_outputs(root_dir)`.
- **A job you ran via the lifecycle** → `dft_fetch_results(job_id)` already wraps
  the summarizer and adds provenance.

## What the summary contains
- normal-termination flag and inferred job type
- final single-point energy (Hartree)
- HOMO/LUMO indices and energies, HOMO–LUMO gap (eV)
- imaginary-frequency count (for `freq` jobs)
- a `warnings` list of reliability signals — **read these, don't ignore them**

## Reliability checklist
1. **Normal termination?** No `ORCA TERMINATED NORMALLY` → the run died; treat
   any parsed numbers as suspect and triage (see failure-triage.md).
2. **Converged?** For an optimization, confirm the geometry converged
   (`HURRAY`/convergence block). A missing convergence marker means *do not*
   present the geometry/energy as final.
3. **Minimum verified?** A bare `Opt` is not proof of a minimum. Require a `freq`
   result with **zero imaginary frequencies**. Exactly one imaginary frequency =
   a first-order saddle point (transition state), not a minimum — useful if
   that's the goal, misleading otherwise. Small imaginary modes (< ~i50 cm⁻¹)
   are often numerical; consider a tighter grid / re-optimization.
4. **Open-shell caveats?** UHF/UKS outputs may carry spin-contamination; treat
   frontier-orbital (HOMO/LUMO) assignments cautiously and check ⟨S²⟩ against the
   expected value. The summarizer flags open-shell markers — propagate that
   caution rather than over-stating orbital interpretations.
5. **Multiple orbital blocks?** HOMO/LUMO parsing can be heuristic for
   open-shell/multi-block outputs — note reduced confidence.

## How to handle warnings
Warnings are reliability signals, not cosmetic text. If warnings exist:
- still report values that were successfully extracted,
- explicitly downgrade confidence where the warning bears on the result,
- never present an uncertain value as definitive.

## Mapping to chemistry
- **Final energy** — only comparable between calculations with the *same*
  method, basis, dispersion, and solvation. Reaction/binding energies =
  differences of consistently-computed energies.
- **HOMO–LUMO gap** — a qualitative reactivity/optical descriptor; for real
  excitation energies use TD-DFT (Tier 3), not the orbital gap.
- **Thermochemistry** — from a `freq` job (ZPE, enthalpy, Gibbs free energy);
  only valid at a true minimum.
- **Dipole / population analysis** — from the single-point output.

## Closing the loop
Return: the observable + units, a reliability statement (terminated normally?
converged? minimum verified? open-shell caveats?), the verdict vs. any ML
prediction being checked, and provenance (`job_id`/path, method, basis,
solvation). Cache via `ase_store_result`. If the user also wants orbital/density
pictures, proceed to cube-generation.md.
