# VASP Calculation Design

How to choose the calculation type and parameters, and express them through the
`dft_prepare_calculation` `overrides` argument. The tool builds on a pymatgen
input set (`MPRelaxSet` for relax, `MPStaticSet` for static), so you specify
only what should differ from those defaults.

## Contents
- [Calculation-type matrix](#calculation-type-matrix)
- [INCAR decisions the defaults can't make for you](#incar-decisions)
- [Functionals](#functionals)
- [Pseudopotentials (POTCAR)](#pseudopotentials)
- [Parallelization](#parallelization)
- [Overrides cookbook per calc type](#overrides-cookbook)

---

## Calculation-type matrix

| Type | Tier | How to run | Notes |
|------|------|-----------|-------|
| `relax` | 1 | `calc_type="relax"` | Full geometry optimization (`ISIF=3` relaxes cell+ions by default). The workhorse. |
| `static` / `single_point` | 1 | `calc_type="static"` | Single SCF at fixed geometry. Use after a relax for accurate energy/DOS. |
| DOS / PDOS | 2 | `static` + `overrides` | `{"incar": {"NEDOS": 3001, "LORBIT": 11, "ISMEAR": -5}}`, denser k-mesh. |
| Magnetism | 2 | `relax`/`static` + `overrides` | `{"incar": {"ISPIN": 2, "MAGMOM": "..."}}`. See [magnetism](#magnetism). |
| Hubbard +U | 2 | `+ overrides` | `{"incar": {"LDAU": true, "LDAUTYPE": 2, "LDAUL": "...", "LDAUU": "..."}}`. |
| Dielectric/optical | 2 | `static` + `overrides` | `{"incar": {"LOPTICS": true, "NBANDS": <2-3x>, "CSHIFT": 0.1}}`. |
| Spin-orbit | 2 | `+ overrides` | `{"incar": {"LSORBIT": true}}` — needs `vasp_ncl`; set the command in config. |
| Band structure | 3 | **not yet runnable** | Needs line-mode KPOINTS along a k-path; adapter doesn't emit it. Advise: static to get CHGCAR, then non-SCF (`ICHARG=11`) along the path. |
| Phonons | 3 | **not yet runnable** | Finite-displacement (phonopy supercells) or DFPT (`IBRION=8`). Needs supercell generation. |
| AIMD | 3 | **not yet runnable** | `IBRION=0` MD; needs ensemble/thermostat handling. |
| NEB | 3 | **not yet runnable** | Multi-image; needs image interpolation + `IMAGES`. |
| Elastic constants | 3 | **not yet runnable** | `IBRION=6, ISIF=3`; needs strain post-processing. |

For Tier 3, give the user the correct recipe but state plainly it can't be
submitted through the current tools — extending `core/dft/engines/vasp.py` is
the path. Do **not** submit a job hoping it works: an unsupported `calc_type`
falls back to `relax` with a loud warning, which is not what was asked.

---

## INCAR decisions

The pymatgen defaults are reasonable, but these choices are physics-dependent
and you must make them consciously.

### Smearing — `ISMEAR` / `SIGMA`
The single most common source of wrong energies.
- **Metals / unknown:** `ISMEAR=1` (Methfessel-Paxton) or `0` (Gaussian),
  `SIGMA=0.1–0.2`. Check that the entropy term `T*S` per atom is < ~1 meV.
- **Semiconductors / insulators:** `ISMEAR=0`, `SIGMA=0.05`.
- **Accurate DOS / final static of an insulator:** `ISMEAR=-5` (tetrahedron +
  Blöchl), requires a Γ-centered mesh with ≥3–4 k-points and is unsuitable for
  relaxation forces.

### Magnetism — `ISPIN` / `MAGMOM` {#magnetism}
- Turn on spin (`ISPIN=2`) for any system with 3d/4f elements, O₂, radicals, or
  suspected magnetic ordering. Non-spin-polarized runs silently miss the
  magnetic ground state.
- Set initial `MAGMOM` per atom: ~5 for high-spin Fe/Mn/Co/Ni, ~0.6 for most
  others. Good initial moments matter for converging to the right state.
- For antiferromagnetic/ferrimagnetic orders, set signs explicitly and consider
  multiple orderings.

### Hubbard U — `LDAU`
For correlated 3d transition-metal oxides/fluorides and lanthanides, semilocal
DFT under-localizes d/f electrons. Apply DFT+U (`LDAUTYPE=2`, Dudarev) with
literature/MP U values (e.g. Fe 4–5.3, Mn 3.9, Ni 6, V 3.25 eV). Be consistent:
formation energies are only comparable between calculations using the *same* U
and the *same* reference scheme (MP uses a mixed GGA/GGA+U correction).

### Cell vs. ions — `ISIF`
- `ISIF=3`: relax ions + cell shape + volume (default for `relax`; correct for
  bulk equilibrium).
- `ISIF=2`: relax ions only, fixed cell (surfaces, substrate-constrained,
  defects in a fixed host).
- A cell-relaxation at fixed `ENCUT` suffers Pulay stress — converge `ENCUT`
  well or re-relax at the final volume.

### Accuracy / convergence — `PREC`, `EDIFF`, `EDIFFG`, `NELM`, `ALGO`
- `PREC=Accurate`, `EDIFF=1e-5` (1e-6 for phonons/elastic), `EDIFFG=-0.01 to
  -0.02` eV/Å for force convergence.
- `ALGO`: `Normal` (default), `Fast`/`VeryFast` for big systems, `All`/`Damped`
  for hard-to-converge metals/magnets.

---

## Functionals
- **PBE** — default workhorse for geometries/energies.
- **PBEsol** — better lattice constants for solids.
- **SCAN / r²SCAN** — improved energetics, pricier, harder to converge.
- **HSE06** — hybrid, for band gaps; ~10–100× cost, use only on small cells
  after a PBE relax.
Set via the input-set `xc` / appropriate INCAR tags through `overrides`.

## Pseudopotentials {#pseudopotentials}
POTCAR generation requires the pseudopotential library on the cluster. Set
`vasp_pp_path` in config (or `overrides.vasp_pp_path`); otherwise the tool
writes INCAR/POSCAR/KPOINTS and warns that POTCAR is missing — that job cannot
run until POTCAR exists. Prefer the VASP-recommended PBE potentials (e.g.
`Fe_pv`, `O`, `Li_sv`) — pymatgen's default set chooses these.

## Parallelization
- `NCORE` = cores working on one orbital (4–8 typical); `NCORE × KPAR` should
  divide total MPI ranks.
- `KPAR` = k-point groups; helpful when many k-points.
- Set `ntasks` in the submit `resources` to match the node layout, and
  `NCORE`/`KPAR` accordingly via `overrides.incar`.

---

## Overrides cookbook

**Insulator relaxation (e.g. an oxide):**
```json
{"encut": 520, "kpts": 64, "incar": {"ISMEAR": 0, "SIGMA": 0.05, "ISPIN": 2}}
```

**Metal static for accurate energy:**
```json
{"encut": 500, "kpts": 100, "incar": {"ISMEAR": 1, "SIGMA": 0.2}}
```

**Magnetic oxide with +U (e.g. LiFePO₄-like):**
```json
{"incar": {"ISPIN": 2, "MAGMOM": "...per-atom...",
            "LDAU": true, "LDAUTYPE": 2, "LDAUL": "2 -1 -1", "LDAUU": "5.3 0 0"}}
```

**DOS (run as a `static` after relax):**
```json
{"kpts": 200, "incar": {"ISMEAR": -5, "LORBIT": 11, "NEDOS": 3001, "ICHARG": 11}}
```

Always echo the resolved INCAR (`resolved_params.incar`) back to the user at the
plan/confirm gate so the actual settings — not your intent — are what gets
approved.
