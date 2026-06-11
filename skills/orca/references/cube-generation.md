# ORCA Cube Generation (Guarded Workflow)

Generating volumetric cube files for visualization — HOMO/LUMO orbitals,
electron density, and electrostatic potential (ESP) — by driving `orca_plot`
through the `orca_*` tools. This is a **guarded** workflow: `orca_plot` is
version-sensitive, needs a writable directory and matched `.out`/`.gbw` files,
and can silently produce wrong/missing cubes. Always preflight before
generating, and surface reliability caveats with the results.

## Step 0 — Triage gate (always first)
Before any cube generation, confirm the directory is cube-ready:

1. `orca_validate_environment(test_dir=calc_dir)` — checks that `orca_plot` is
   found and the directory is writable.
2. `orca_validate_calc_dir(calc_dir)` — checks for `.out`/`.gbw` files and
   same-stem pairing.

Combine into a readiness judgment and **stop** if any of these hold:
- `orca_plot` not available, or directory not writable → cube generation
  cannot proceed (analysis may still be possible — see output-analysis.md).
- no `.gbw` file present → nothing to plot from.
- multiple `.out`/`.gbw` files with no clear same-stem pair → ambiguous;
  ask the user which calculation, or pass an explicit selection rather than
  guessing.

Treat the validators' warnings as risk signals, not noise. Only continue when
the directory is genuinely cube-ready (or the user has resolved the ambiguity).

## HOMO/LUMO cubes
`orca_generate_homo_lumo_cubes(calc_dir, preference, ngrid, operator)`:
- `preference` — which calculation to use: `"optimization"`, `"single_point"`,
  or `"auto"`.
- `ngrid` — grid intervals as a string, e.g. `"80 80 80"`. Higher = smoother but
  larger/slower; 80³ is a good default, 40³ for quick looks.
- `operator` — spin channel: `0` for alpha/closed-shell, `1` for beta. For
  closed-shell systems use `0`.

**Open-shell caution:** for UHF/UKS the HOMO/LUMO assignment can require manual
verification, and alpha/beta channels differ — generate the relevant `operator`
and state the caveat. Don't present the orbital assignment as definitive when
the summary flagged open-shell markers.

## Electron-density & ESP cubes
`orca_generate_density_esp_cubes(calc_dir, preference, ngrid)`:
- Produces a **matched pair** (density + ESP) intended to be used together
  (e.g. ESP mapped onto a density isosurface).
- The tool validates **grid consistency** between the two cubes. If they do not
  share a consistent grid, do **not** treat them as a valid pair for
  ESP-on-density visualization — report the inconsistency.

## Specific orbital
`orca_generate_mo_cube(...)` is the lower-level tool for an arbitrary MO index;
prefer `orca_generate_homo_lumo_cubes` unless a specific non-frontier orbital is
requested.

## Reliability rules for all cube generation
If warnings are present after generation:
- report the cube paths if generation succeeded,
- state reduced confidence clearly (ambiguous `.gbw` match → cube may not
  correspond to the intended calculation; `orca_plot` menu-version signals →
  outputs may be inconsistent; grid inconsistency → density/ESP not a valid
  pair),
- never present an unvalidated/ambiguous cube as definitively correct.

## Reporting
Return the generated cube file paths, the selected `.out`/`.gbw`, the grid used,
the spin operator (for orbitals), grid-consistency status (for density/ESP), and
the reliability caveats — so the user knows exactly what was visualized and how
much to trust it.
