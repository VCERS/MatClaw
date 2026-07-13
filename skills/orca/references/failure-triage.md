# ORCA Failure Triage

When `dft_get_calculation_status` reports `failed`, or a job terminates without
`ORCA TERMINATED NORMALLY`, diagnose before resubmitting. Inspect the job's
`orca.out` and the SLURM error file via the workdir paths, match the symptom,
fix through `dft_restart_calculation` (which copies the `.gbw` for an
initial-guess restart), and re-run the plan/confirm gate.

## SCF non-convergence
**Symptom:** SCF fails to converge within the iteration limit; energy
oscillates.
**Fixes (in order):**
- Add `SlowConv` (or `VerySlowConv` for hard cases) to `keywords`.
- Switch the converger: `%scf maxiter 300 end`, or try `KDIIS` / `SOSCF`.
- Improve the initial guess: a better starting geometry, or restart from a
  related `.gbw` (`dft_restart_calculation` copies it; add `! MOREAD` and
  `%moinp "orca.gbw"` to actually read it).
- Use a smaller/cleaner basis to get a guess, then step up.
- For metals/diffuse cases, level-shifting (`%scf shift ... end`) can help.

## Geometry-optimization trouble
**Symptom:** optimization doesn't converge, oscillates, or blows up.
**Fixes:**
- Restart from the last geometry/`.gbw`.
- Tighten/relax convergence (`TightOpt` vs. default) and check `%geom maxiter`.
- Pre-clean the starting geometry; remove obviously bad contacts.
- Add/redefine internal coordinates or constraints for floppy systems.
- Confirm charge/multiplicity are correct — a wrong spin state often manifests
  as pathological optimization.

## Open-shell / spin issues
**Symptom:** high spin-contamination (⟨S²⟩ far from S(S+1)), or convergence to
the wrong state.
**Fixes:**
- Verify multiplicity is physically correct.
- Provide a sensible initial guess; consider broken-symmetry (`%scf
  brokensym ... end`) for antiferromagnetic coupling.
- Use a range-separated hybrid (e.g. `wB97X-D4`) which is often better behaved.
- Treat HOMO/LUMO and orbital pictures with caution and note the caveat.

## Resource / runtime faults
**Symptom:** SLURM `TIMEOUT`, out-of-memory, or a crash.
**Fixes:**
- Memory: set `%maxcore <MB-per-core>` to a realistic value (ORCA needs it);
  request more memory or fewer cores per node.
- Walltime: request more, or split (e.g. optimize first, frequencies as a
  separate job).
- `nprocs` mismatch: ensure `overrides.nprocs` matches the submitted `ntasks`.

## Restart mechanics
`dft_restart_calculation(job_id, overrides=...)` clones the job into a fresh
workdir, copies `orca.inp` and `orca.gbw`, and returns a new `job_id` in state
`prepared`. To actually reuse the wavefunction guess, add `! MOREAD` and
`%moinp "orca.gbw"` via `overrides.keywords`/`overrides.blocks`. Apply your fix,
then plan/confirm → submit. Always tell the user what failed and what you
changed — never restart silently.
