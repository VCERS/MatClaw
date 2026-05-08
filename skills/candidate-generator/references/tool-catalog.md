# Detailed Tool Catalog

Complete specifications for all candidate generation MCP tools.

## Phase 0: Composition Discovery

### `composition_enumerator` — Oxidation-Balanced Enumeration
Generates all charge-balanced compositions from element lists and oxidation state constraints.
Use when you know which elements to explore but not which compositions exist.

**Key parameters:**
- `elements`: list of element symbols, e.g. `['Li', 'Mn', 'P', 'O']`
- `oxidation_states`: dict mapping elements to allowed oxidation states, e.g. `{'Li': [1], 'Mn': [2, 3], 'P': [5], 'O': [-2]}`
- `max_formula_units`: cap on formula unit count (default 6, increase for complex compositions)
- `max_atoms_per_formula`: hard limit on total atoms (default 30, prevents combinatorial explosion)
- `anion_cation_ratio_max`: maximum anion:cation ratio (default 4.0, excludes nonsense stoichiometries)
- `min_cation_fraction`: minimum cation fraction (default 0.05, excludes trace compositions)
- `require_all_elements`: if True, only returns compositions containing ALL specified elements (default True)
- `allow_mixed_valence`: if True, allows mixed oxidation states (default True)
- `sort_by`: `'atoms'` (fewest atoms first), `'anion_ratio'`, `'alphabetical'`
- `output_format`: `'minimal'` (formula strings) or `'detailed'` (full metadata)

**Returns:**
```python
{
  "success": True,
  "count": 12,
  "compositions": ["Li3PO4", "LiMnPO4", ...]
  # OR with output_format='detailed':
  "compositions": [
    {
      "formula": "LiMnPO4",
      "num_atoms": 7,
      "anion_cation_ratio": 1.33,
      "oxidation_states": {"Li": 1, "Mn": 2, "P": 5, "O": -2}
    }
  ]
}
```

**Chemical filters:**
- `max_atoms_per_formula=30`: Prevents unrealistically large formulas
- `anion_cation_ratio_max=4.0`: Prevents anion-heavy compositions
- `min_cation_fraction=0.05`: Prevents trace cation compositions

---

### `pymatgen_substitution_predictor` — ICSD-Based Substitution
Predicts likely element substitutions using data mining from 100k+ ICSD structures.

**Key parameters:**
- `composition`: starting composition, e.g. `'LiFePO4'`
- `to_this_composition`: if False (default), finds what this composition can become
- `threshold`: probability cutoff (0.001 = permissive, 0.1 = strict)
- `max_suggestions`: limit number of suggestions
- `group_by_probability`: if True, returns {high: [...], medium: [...], low: [...]}

**Returns:**
```python
{
  "suggestions": {
    "high": [{"formula": "LiMnPO4", "probability": 0.85}],
    "medium": [{"formula": "LiCoPO4", "probability": 0.45}]
  }
}
```

**Limitation:** ICSD patterns are conservative (based on existing materials only).

---

### `mp_search_materials` — Template Structure Search
Queries Materials Project for structures matching composition/chemistry constraints.

**Key parameters for template search:**
- `elements`: list of elements, e.g. `['Li', 'Fe', 'P', 'O']`
- `num_elements`: constrain to binary (2), ternary (3), quaternary (4), etc.
- `crystal_system`: `'cubic'`, `'tetragonal'`, `'orthorhombic'`, etc.
- `spacegroup_number`: specific space group
- `is_stable`: True (only thermodynamically stable)
- `limit`: max results (default 100)

---

## Phase 1: Seed Structure

### `pymatgen_prototype_builder` — Build from Spacegroup
Builds an ideal crystal from spacegroup number/symbol, species list, and lattice parameters.

**Key parameters:**
- `spacegroup`: int (1–230) or Hermann-Mauguin symbol, e.g. `225` or `"Fm-3m"`
- `species`: list of element symbols
- `lattice_parameters`: `[a, b, c, alpha, beta, gamma]` in Å and degrees; `[a]` for cubic
- `wyckoff_positions`: optional dict mapping Wyckoff labels to species
- `output_format`: `'dict'` (default), `'poscar'`, `'cif'`, `'ase'`

**Common prototypes:**

| Prototype | SG # | Symbol | Example |
|-----------|------|--------|---------|
| Rock-salt | 225 | Fm-3m | NaCl, LiF, MgO |
| Perovskite | 221 | Pm-3m | BaTiO₃, SrTiO₃ |
| Spinel | 227 | Fd-3m | MgAl₂O₄, LiMn₂O₄ |
| Layered oxide | 166 | R-3m | LiCoO₂, LiNiO₂ |
| Olivine | 62 | Pnma | LiFePO₄, LiMnPO₄ |
| Rutile | 136 | P4₂/mnm | TiO₂, SnO₂ |
| Wurtzite | 186 | P6₃mc | ZnO, GaN |
| Fluorite | 225 | Fm-3m | CaF₂, CeO₂ |

---

## Phase 2: Chemical Space Exploration

### `pymatgen_substitution_generator` — Ordered Enumeration
Replaces elements by FULLY replacing specific sites, generating ORDERED structures with integer occupancy.

**CRITICAL: Creates ORDERED structures, NOT fractional occupancy.**
- `fraction=0.15` means "fully replace 15% of sites" (integer occupancy on those sites)
- For fractional occupancy (80% Ni + 20% Mn on same site), use `pymatgen_disorder_generator`

**Key parameters:**
- `substitutions`: 
  - `{'Li': 'Na'}` — full swap
  - `{'Li': ['Na', 'K']}` — one variant per replacement
  - `{'Li': {'replace_with': 'Na', 'fraction': 0.5}}` — FULLY replace 50% of Li sites
- `n_structures`: variants per substitution combination (default 5)
  - Set to **1** for deterministic full swaps (`fraction=1.0`)
  - Higher values for partial replacement generate different random orderings
- `max_attempts`: **hard cap on total output** (default 50)
  - Set `max_attempts ≥ n_structures × num_combinations` to avoid truncation
- `enforce_charge_neutrality`: set True for ionic materials
- `output_format`: `'ase'` when feeding to `ase_store_result`

---

### `pymatgen_ion_exchange_generator` — Charge-Neutral Substitution
Replaces a mobile ion (e.g. Li⁺) with one or more ions, automatically adjusting stoichiometry for charge neutrality.

**Key parameters:**
- `replace_ion`: element to replace, e.g. `'Li'`
- `with_ions`: `['Na', 'K']` (equal weight) or `{'Na': 0.6, 'Mg': 0.4}` (weighted)
- `exchange_fraction`: fraction of sites to exchange (0–1), default `1.0`
- `allow_oxidation_state_change`: `False` (default) = only neutral structures
- `max_structures`: cap on returned structures (default 10)

**Use for:** Battery cathode analogues (Li → Na/K), charge-balanced doping (Ca²⁺ → La³⁺)

---

## Phase 3: Disorder Generation & Resolution

### `pymatgen_disorder_generator` — Add Fractional Occupancy
**REQUIRED TOOL FOR FRACTIONAL SITE OCCUPANCY**

Converts ordered structures into disordered structures with FRACTIONAL site occupancies.
This is the ONLY tool for creating partial substitution materials like Li[Ni₀.₈Mn₀.₂]O₂.

**Creates FRACTIONAL OCCUPANCY, not ordered enumeration:**
- Creates sites with partial occupancy (e.g., 80% Ni + 20% Mn on same site)
- Output: Single disordered structure per input
- For ordered configurations, use `pymatgen_substitution_generator`

**Key parameters:**
- `site_substitutions`: dict mapping elements to fractional occupancies
  - Format: `{element: {species1: fraction1, species2: fraction2}}`
  - Binary: `{'Ni': {'Ni': 0.8, 'Mn': 0.2}}` → Li[Ni₀.₈Mn₀.₂]O₂
  - Ternary: `{'Co': {'Ni': 0.333, 'Mn': 0.333, 'Co': 0.334}}` → NMC
  - Fractions must sum to 1.0
- `site_selector`: which sites receive disorder (default: `'all_equivalent'`)
- `validate_charge_neutrality`: `True` (default) — warns if charge imbalance
- `composition_tolerance`: tolerance for fraction sums (default 0.01)

**Typical workflow:**
1. Get ordered structure (MP or prototype)
2. Add disorder with `disorder_generator`
3. Generate SQS with `pymatgen_sqs_orderer` for DFT

---

### `pymatgen_order_enumerator` — Exhaustive Ordering
Takes structures with fractional site occupancies and returns all symmetry-inequivalent ordered supercell approximants.

**Key parameters:**
- `supercell_size`: supercell multiplier (1–4, default 2)
  - Keep ≤ 2 for ternary systems (combinatorial explosion)
- `n_structures`: max ordered structures returned (default 20, max 500)
- `sort_by`: `'ewald'` (default, lowest energy first), `'num_sites'`, `'random'`
- `add_oxidation_states`: auto-assign for Ewald ranking (default True)
- `refine_structure`: re-symmetrize before enumeration (default True)

**Use when:** Need complete ordered-configuration space, ground-state search, cluster expansion training.

---

### `pymatgen_sqs_orderer` — Special Quasirandom Structures
Finds ordered supercell whose pair correlations best mimic a perfectly random alloy.

**Key parameters:**
- `supercell_size`: target formula units (default 8; use 8–16 for binary, 12–24 for ternary)
- `supercell_matrix`: explicit `[nx, ny, nz]` or 3×3 matrix
- `n_structures`: independent SQS candidates (default 3)
- `n_mc_steps`: Monte Carlo steps (default 50,000; increase for multicomponent)
- `n_shells`: correlation shells (default 4)
- `seed`: for reproducibility
- `use_mcsqs`: use ATAT `mcsqs` binary if available

**Use when:** Modeling solid solutions, high-entropy materials where disorder is the physical state.

---

## Phase 4: Defects

### `pymatgen_defect_generator` — Point Defect Supercells
Takes perfect bulk host structure and generates one supercell per symmetry-inequivalent defect site.

**Key parameters:**
- `vacancy_species`: `['Li', 'O']` — generate V_Li, V_O defects
- `substitution_species`: `{'Fe': ['Mn', 'Co']}` — Mn_Fe and Co_Fe substitutionals
- `interstitial_species`: `['Li']` — find void sites and insert
- `charge_states`: `{'V_Li': [-1, 0, 1]}` — metadata only (structures are neutral)
- `supercell_min_atoms`: target atoms in defect supercell (default 64)
- `inequivalent_only`: True (default) — only symmetry-distinct defects

---

## Phase 5: Perturbation

### `pymatgen_perturbation_generator` — Structural Ensemble
Applies random atomic displacements and/or lattice strain to create perturbed ensembles.

**Key parameters:**
- `displacement_max`: max displacement per atom in Å (default 0.1; range 0.05–0.2)
- `strain_percent`: 
  - `None` — off
  - scalar — uniform strain
  - `[min, max]` — random range
  - 6-element Voigt tensor
- `n_structures`: perturbed copies per input (default 10, max 200)
- `seed`: for reproducibility

**Use for:**
- Break symmetry before DFT (avoid saddle points)
- ML dataset augmentation
- Elastic property screening
