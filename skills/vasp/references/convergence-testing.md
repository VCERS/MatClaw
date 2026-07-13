# ENCUT & k-point Convergence Testing

Absolute and relative DFT energies are only meaningful once the plane-wave
cutoff (`ENCUT`) and k-point density are converged. Skipping this is the most
common way to produce confident-but-wrong numbers. Do it before any production
calculation where energy differences matter (formation energies, reaction
energies, EOS, defect formation). For rough screening you may skip it — but then
explicitly label the result as unconverged/qualitative.

## Why both, and why separately
ENCUT controls basis-set completeness; k-mesh controls Brillouin-zone sampling.
They are independent error sources, so converge them one at a time, holding the
other fixed at a generous value.

## Protocol

This is a small fan of cheap **static** jobs on the (pre-relaxed) structure. It
is the canonical use case for the plan/confirm gate: **propose the full sweep
matrix and its total job count/cost, get approval, then submit as a batch.**

**Step 1 — ENCUT sweep** (fix a dense k-mesh, e.g. reciprocal density 100):
- Sweep `ENCUT` ∈ {0.8, 1.0, 1.3} × `ENMAX` of the hardest POTCAR — concretely
  often {400, 450, 500, 550, 600} eV.
- Each is `dft_prepare_calculation(engine="vasp", calc_type="static",
  overrides={"encut": E, "kpts": 100})` → submit as a batch.

**Step 2 — k-point sweep** (fix ENCUT at the converged value from step 1):
- Sweep reciprocal density ∈ {30, 50, 80, 120, 160} (or explicit grids
  `[2,2,2] … [8,8,8]`).
- `overrides={"encut": <converged>, "kpts": density}`.

**Step 3 — read the trend.** `dft_fetch_results` for each; plot/inspect
`final_energy_eV` **per atom** vs. the swept parameter.

## Tolerances
- Converged when the energy/atom changes by **< 1 meV/atom** (tight; for many
  screening tasks **< 5 meV/atom** is acceptable — state which you used).
- Choose the smallest ENCUT / coarsest k-mesh meeting the tolerance, then use
  those for production. For elastic/phonon work, tighten to ENCUT ≥ 1.3×ENMAX.

## Practical notes
- Always use the **same** structure (the ML- or DFT-pre-relaxed one) for every
  point, so you isolate the parameter effect.
- Metals need denser k-meshes than insulators — expect k-convergence to dominate.
- Cell relaxations have Pulay stress at finite ENCUT; converge ENCUT generously
  or re-relax at the converged-volume cell.
- Report the converged values and the achieved tolerance as part of provenance —
  downstream comparisons are only valid between calculations sharing them.
