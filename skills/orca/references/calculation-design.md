# ORCA Calculation Design

How to choose method, basis, and settings, and express them through
`dft_prepare_calculation` `overrides`. The tool's default is `B3LYP/def2-SVP`;
you specify only what should differ.

## Contents
- [Charge & multiplicity](#charge--multiplicity)
- [Calculation-type matrix](#calculation-type-matrix)
- [Methods (functionals)](#methods)
- [Basis sets](#basis-sets)
- [Dispersion, acceleration, accuracy](#dispersion-acceleration-accuracy)
- [Solvation](#solvation)
- [Overrides cookbook](#overrides-cookbook)

---

## Charge & multiplicity
Get these right before anything else — a wrong spin state gives a precise wrong
answer. Multiplicity = 2S+1:
- closed-shell neutral molecule → charge 0, mult 1
- radical / odd electron count → mult 2 (doublet)
- O₂, carbenes, many transition-metal centers → mult 3+ (triplet/higher)
Cross-check: (electron count parity) must be consistent with multiplicity
(even electrons → odd multiplicity, odd electrons → even multiplicity). If the
ground-state spin is unknown, compare energies of plausible multiplicities.

## Calculation-type matrix

| Type | Tier | How to run | Notes |
|------|------|-----------|-------|
| `single_point` | 1 | `calc_type="single_point"` | Energy/properties at a fixed geometry. |
| `opt` | 1 | `calc_type="opt"` | Geometry optimization. **Does not prove a minimum** — follow with `freq`. |
| `freq` | 1 | `calc_type="freq"` | Hessian → IR, thermochemistry, minimum/TS check. |
| `opt_freq` | 1 | `calc_type="opt_freq"` | Optimize then frequencies in one job — the standard "is this a real minimum" workflow. |
| Solvation | 2 | `+ overrides.blocks` | `%cpcm` / `SMD` — see [Solvation](#solvation). |
| Scan (PES) | 2 | `opt` + `%geom Scan` block | Relaxed scan along a coordinate. |
| Constrained opt | 2 | `opt` + `%geom Constraints` | Freeze bonds/angles. |
| TD-DFT / excited states | 3 | **not yet runnable** | Needs a `%tddft` block + roots handling; advise but don't submit. |
| NEB-TS | 3 | **not yet runnable** | Reactant/product images; multi-structure input. |
| CASSCF / NEVPT2 | 3 | **not yet runnable** | Active-space selection; specialized blocks. |
| NMR / polarizability | 3 | **not yet runnable** | Dedicated property blocks. |

For Tier 3, give the correct ORCA recipe but be explicit it isn't executable
through the current tools (extend `core/dft/engines/orca.py`). An unsupported
`calc_type` falls back to `single_point` with a warning.

## Methods
- **Hybrids:** `B3LYP` (ubiquitous), `PBE0`, `wB97X-V`/`wB97X-D4` (range-
  separated, excellent general accuracy). Hybrids need `RIJCOSX` for speed.
- **Composite "3c" methods:** `r2SCAN-3c` (great cost/accuracy for geometries
  and reaction energies), `B97-3c`, `PBEh-3c` — these bundle basis + dispersion,
  so use them *without* a separate basis/dispersion.
- **GGA/meta-GGA:** `BP86`, `TPSS` — cheaper, for large systems or initial
  guesses.
Pair non-composite functionals with dispersion (below).

## Basis sets
The Karlsruhe `def2` family is the default choice:
- `def2-SVP` — fast, screening, large systems.
- `def2-TZVP` — production accuracy for most properties.
- `def2-QZVP` — benchmark/extrapolation.
- Add diffuse functions (`def2-TZVPD`/`def2-SVPD`) for **anions, excited states,
  polarizabilities, and noncovalent interactions**.
- For heavy elements (Z>36) `def2` uses effective core potentials automatically.

## Dispersion, acceleration, accuracy
- **Dispersion:** add `D4` (preferred) or `D3BJ` to any non-composite
  functional — essential for geometries, conformers, and binding energies.
- **`RIJCOSX`:** the RI-J + chain-of-spheres exchange approximation; ~order-of-
  magnitude speedup for hybrids at negligible accuracy cost. Add an auxiliary
  basis (`def2/J`) — ORCA picks it automatically for `def2` orbital bases.
- **`TightSCF`:** tighten SCF convergence for reliable energies/gradients.
- **`DEFGRID3`:** denser integration grid for meta-GGAs / tricky cases.

## Solvation
Implicit solvent via a `%`-block in `overrides.blocks`:
- **CPCM:** `"%cpcm epsilon 78.4 refrac 1.33 end"` or by name.
- **SMD** (better for solvation free energies):
  `"%cpcm smd true SMDsolvent \"water\" end"`.
Use for any condensed-phase property; gas-phase otherwise.

## Overrides cookbook

**Production optimization + frequencies of a neutral organic:**
```json
{"method": "wB97X-D4", "basis": "def2-TZVP", "keywords": "RIJCOSX TightSCF"}
```
with `calc_type="opt_freq"`, `charge=0`, `multiplicity=1`.

**Fast screening single-point:**
```json
{"method": "r2SCAN-3c", "keywords": "TightSCF"}
```
(composite method — omit `basis` and dispersion; they're built in.)

**Anion in water:**
```json
{"method": "wB97X-D4", "basis": "def2-TZVPD",
 "keywords": "RIJCOSX TightSCF",
 "blocks": ["%cpcm smd true SMDsolvent \"water\" end"]}
```
with the correct negative `charge`.

**Open-shell radical (doublet):** as above with `multiplicity=2`; expect UHF and
read the open-shell caveats in output-analysis / failure-triage.

Always echo the resolved `! ...` line (from `resolved_params`) to the user at
the plan/confirm gate so the actual input — not your intent — is approved.
