# Detailed Phase Guides

Complete detailed instructions for each workflow phase in the candidate generation pipeline.

---

## Phase 0: Composition Discovery

**When to use:** You only know elements (e.g., "Li-Mn-P-O"), not specific compositions

**Skip if:** You already have target composition or structure

### Three Strategies

#### 1. Exhaustive Enumeration

**Tool:** `composition_enumerator`

**Best for:**
- Exploratory discovery in well-defined chemical space
- Systematic coverage of all possible stoichiometries
- Small element sets (≤4 elements)

**Algorithm:**
```python
composition_enumerator(
    elements=['Li', 'Mn', 'P', 'O'],
    oxidation_states={
        'Li': [1],
        'Mn': [2, 3, 4],
        'P': [5],
        'O': [-2]
    },
    max_formula_units=8,  # Balance between coverage and size
    output_format='json'
)
```

**Output:** 50-500 charge-balanced compositions

#### 2. Template-Based Discovery

**Tool:** `mp_search_materials`

**Best for:**
- Known analogues exist
- Targeting specific structure types
- Quickly identifying stable compositions

**Algorithm:**
```python
# Find all stable Li-Mn-P-O materials in MP
templates = mp_search_materials(
    elements=['Li', 'Mn', 'P', 'O'],
    is_stable=True
)

# Extract composition patterns
patterns = [extract_composition_pattern(mp_struct) for mp_struct in templates]
```

**Output:** 5-20 known stable compositions as templates

#### 3. ICSD Substitution Prediction

**Tool:** `pymatgen_substitution_predictor`

**Best for:**
- Starting from known material
- Element substitution screening
- Probabilistic exploration

**Algorithm:**
```python
# Starting from LiFePO4, find likely substitutions
predictions = pymatgen_substitution_predictor(
    composition='LiFePO4',
    threshold=0.01,  # 1% probability cutoff
    output_count=50
)
```

**Output:** 20-100 predicted compositions with likelihood scores

### Decision Logic

```
Known analogue exists?
└─ YES: Template-based + ICSD (fast, targeted)
└─ NO:
    └─ Well-studied system (Li batteries, oxide perovskites)?
        └─ YES: Template-based first, enumeration to fill gaps
        └─ NO: Exhaustive enumeration + stability filtering
```

### Output Processing

After generating compositions:

1. **Filter by thermodynamic plausibility:**
```python
for comp in compositions:
    mp_result = mp_search_materials(formula=comp)
    if mp_result['count'] > 0:
        comp['mp_exists'] = True
        comp['priority'] = 'high'  # Known stable
```

2. **Check for structures in MP:**
```python
for comp in compositions:
    if comp['mp_exists']:
        # Get structure → skip Phase 1, go to Phase 2
        structures = mp_get_material_properties(...)
    else:
        # Need to build structure → Phase 1
        comp['needs_prototype'] = True
```

---

## Phase 1: Seed Structure Building

**When to use:** Need to build structure from scratch (no MP structure available)

**Skip if:** Structure already exists from MP, CIF, or ASE database

### Common Crystal Prototypes

**Tool:** `pymatgen_prototype_builder`

| Prototype | Spacegroup | Parameters | Example Materials |
|-----------|------------|------------|-------------------|
| Rock-salt | 225 (Fm-3m) | `a` | NaCl, LiF, MgO |
| Perovskite | 221 (Pm-3m) | `a` | BaTiO₃, SrTiO₃, CaTiO₃ |
| Spinel | 227 (Fd-3m) | `a` | MgAl₂O₄, LiMn₂O₄ |
| Layered oxide | 166 (R-3m) | `a`, `c` | LiCoO₂, LiNiO₂ |
| Olivine | 62 (Pnma) | `a`, `b`, `c` | LiFePO₄, LiMnPO₄ |
| Wurtzite | 186 (P63mc) | `a`, `c` | ZnO, GaN |
| Fluorite | 225 (Fm-3m) | `a` | CaF₂, UO₂ |

### Parameter Estimation

When you don't know lattice parameters:

1. **Volume-based estimation:**
```python
from pymatgen.core import Composition
comp = Composition("LiFeO2")
estimated_volume = sum(element.atomic_volume for element in comp) * 1.2  # Add packing factor
```

2. **Ionic radius estimation:**
```python
from pymatgen.core import Species
Li_radius = Species("Li", 1).ionic_radius
Fe_radius = Species("Fe", 3).ionic_radius
O_radius = Species("O", -2).ionic_radius

# For rock-salt: a ≈ 2 * (cation_radius + anion_radius)
a = 2 * (Li_radius + O_radius)
```

3. **Database lookup from similar:**
```python
# Find similar material in MP
similar = mp_search_materials(elements=['Li', 'Fe', 'O'])
a_estimate = similar[0]['lattice']['a']
```

### Building Example

```python
# Build LiFeO2 rock-salt seed structure
seed = pymatgen_prototype_builder(
    spacegroup=225,  # Fm-3m
    species=['Li', 'Fe', 'O', 'O'],  # Wyckoff positions
    lattice_parameters=[4.2],  # Estimated from ionic radii
    output_format='cif'
)
```

**Next steps:** Phase 2 (chemical exploration)

---

## Phase 2: Chemical Space Exploration

**When to use:** Want to explore different compositions/dopings while keeping structure

**Skip if:** Want to keep exact composition

### Branch A: Ion Exchange Generator

**Tool:** `pymatgen_ion_exchange_generator`

**Best for:**
- Battery materials (Li → Na, Mg)
- Ionic conductors
- Charge-critical materials

**Physical basis:** Maintains charge neutrality by automatically adjusting stoichiometry

**Examples:**

```python
# Example 1: Li → Na battery analogue
ion_exchange_generator(
    input_structures=licoo2_structure,
    replace_ion='Li+',
    with_ions=['Na+'],
    exchange_fraction=1.0,  # Complete replacement
    output_format='cif'
)
# LiCoO2 → NaCoO2 (charge neutral)

# Example 2: Partial Mg doping (charge-balancing)
ion_exchange_generator(
    input_structures=licoo2_structure,
    replace_ion='Li+',
    with_ions=['Mg2+'],
    exchange_fraction=0.5,  # 50% replacement
    output_format='cif'
)
# LiCoO2 → Li₀.₅Mg₀.₂₅CoO2 (automatic stoichiometry adjustment for charge)
```

**Troubleshooting:**
- Returns `count: 0`? Try different `exchange_fraction` values (0.25, 0.5, 0.75, 1.0)
- Multiple ions? Specify `with_ions=['Na+', 'K+']` for alternatives

### Branch B: Substitution Generator

**Tool:** `pymatgen_substitution_generator`

**Best for:**
- Isostructural analogue screening
- ML training set generation
- Exploratory screening where charge is handled post-hoc

**Physical basis:** Deterministic combinatorial element swaps (integer occupancy)

**Examples:**

```python
# Example 1: B-site metal scan in perovskite
substitution_generator(
    input_structures=batio3_structure,
    substitutions={'Ti': ['Zr', 'Hf', 'Sn', 'Ce']},
    n_structures=1,  # One structure per substitution
    max_attempts=4,  # 4 metals × 1 = 4 attempts
    output_format='cif'
)
# Outputs: BaZrO3, BaHfO3, BaSnO3, BaCeO3

# Example 2: Multi-site substitution
substitution_generator(
    input_structures=lini_structure,
    substitutions={
        'Li': ['Na', 'K'],
        'Ni': ['Co', 'Mn']
    },
    n_structures=4,  # 2 Li × 2 TM = 4 combinations
    max_attempts=8,  # 2× for safety
    output_format='cif'
)
# Outputs: NaCoO2, NaMnO2, KCoO2, KMnO2
```

**Parameter calculation:**
```python
n_combinations = product of len(substitution_lists)
max_attempts = n_structures * n_combinations
```

**Troubleshooting:**
- Tool hangs? Increase `max_attempts`
- Duplicate structures? Increase `n_structures` or reduce substitution options

---

## Phase 3: Disorder Resolution

### Creating Disorder: disorder_generator

**Tool:** `pymatgen_disorder_generator`

**Best for:**
- Dilute doping (< 10%)
- Statistical disorder representation
- Fast screening with unit cell

**Physical basis:** Creates fractional site occupancies (e.g., 97% Sr + 3% Sm on same crystallographic site)

**Examples:**

```python
# Example 1: 3% Sm doping in SrNb2O6
disorder_generator(
    input_structures=srnb2o6_structure,
    site_substitutions={'Sr': {'Sm': 0.03, 'Sr': 0.97}},
    output_format='cif'
)
# Output: Sr₀.₉₇Sm₀.₀₃Nb₂O₆ (disordered)

# Example 2: Multi-element doping
disorder_generator(
    input_structures=srnb2o6_structure,
    site_substitutions={
        'Sr': {'Sm': 0.02, 'Eu': 0.01, 'Sr': 0.97}
    },
    output_format='cif'
)
# Output: Sr₀.₉₇Sm₀.₀₂Eu₀.₀₁Nb₂O₆
```

**Critical:** Fractions must sum to 1.0 **per site** (see normalization section below)

### Resolving Disorder: Three Strategies

#### Strategy 1: Majority-Species Orderer

**Tool:** `pymatgen_majority_orderer`

**When to use:** Doping concentration < 10%

**Physical basis:** Dilute limit - host lattice dominates, minority species negligible

```python
majority_orderer(
    input_structures=disordered_cif,
    check_ordered_input=True,
    output_format='cif'
)
# Sr₀.₉₇Sm₀.₀₃Nb₂O₆ → SrNb₂O₆ (Sm removed)
```

**Advantages:** Fast, preserves unit cell, valid for screening  
**Limitations:** Only valid < 10% doping, loses dopant information

#### Strategy 2: Enumeration Orderer

**Tool:** `pymatgen_enumeration_orderer`

**When to use:**
- Site-specific studies (which site gives best properties?)
- Exhaustive configuration search
- Small supercells (< 20 atoms)

**Physical basis:** Enumerates all symmetry-distinct orderings

```python
enumeration_orderer(
    input_structures=disordered_cif,
    supercell_size=2,  # 2×2×2 = 8× unit cell
    n_structures=10,
    sort_by='ewald',  # Electrostatic energy
    output_format='cif'
)
# Li₀.₅Na₀.₅Cl → 10 ordered configurations (e.g., Li₂Na₂Cl₄ variants)
```

**Advantages:** Exhaustive, identifies ground states  
**Limitations:** Combinatorial explosion (keep supercell_size ≤ 2)

#### Strategy 3: SQS Orderer

**Tool:** `pymatgen_sqs_orderer`

**When to use:**
- High-concentration doping (> 20%)
- Solid solutions (random alloys)
- High-entropy materials (≥4 mixing species)

**Physical basis:** Quasi-random structures match correlation functions of true disorder

```python
sqs_orderer(
    input_structures=disordered_cif,
    supercell_size=4,  # 4×4×4 = 64× unit cell
    n_structures=5,
    n_mc_steps=100000,  # Increase for multicomponent
    output_format='cif'
)
# Li[Ni₀.₆Mn₀.₂Co₀.₂]O₂ → 64-atom ordered approximant
```

**Advantages:** Most accurate for concentrated disorder  
**Limitations:** Large supercells (10-50× more atoms), MC convergence

### Fractional Occupancy Normalization

**Problem:** `disorder_generator` requires fractions to sum to **1.0 per site**, not per formula unit.

**For compounds with multiple atoms per site (Bi₂, Nb₂, Y₃), divide by stoichiometric coefficient:**

```python
# ❌ WRONG: BaBi₂(MoO₄)₄ with 3% Sm doping
site_substitutions = {'Bi': {'Sm': 0.03, 'Bi': 1.97}}  # Sums to 2.0 - ERROR!

# ✅ CORRECT: Normalize by stoichiometry (Bi₂ → divide by 2)
site_substitutions = {'Bi': {'Sm': 0.015, 'Bi': 0.985}}  # Sums to 1.0 ✓
```

**Normalization helper:**
```python
def normalize_site_subs(formula, site_subs):
    from pymatgen.core import Composition
    comp = Composition(formula)
    normalized = {}
    for element, occupancies in site_subs.items():
        stoich = comp[element]
        normalized[element] = {
            species: frac / stoich 
            for species, frac in occupancies.items()
        }
    return normalized
```

---

## Phase 4: Defect Generation

**Tool:** `pymatgen_defect_generator`

**When to use:** Point defect supercells (vacancies, substitutions, interstitials)

**Physical basis:** Creates symmetry-inequivalent defect configurations in supercells

### Vacancy Defects

```python
defect_generator(
    input_structure=perfect_host,  # Single structure only!
    vacancy_species=['Li'],
    supercell_min_atoms=64,
    output_format='cif'
)
# Generates Li vacancy supercells for each symmetry-inequivalent site
```

### Substitutional Defects

```python
defect_generator(
    input_structure=licoo2_structure,
    substitution_species={'Co': ['Mn', 'Fe', 'Ni']},
    supercell_min_atoms=64,
    output_format='cif'
)
# Co → Mn/Fe/Ni substitutional defects
```

### Mixed Defects

```python
defect_generator(
    input_structure=host,
    vacancy_species=['Li'],
    substitution_species={'Mn': ['Fe']},
    interstitial_species=['Na'],
    supercell_min_atoms=80,
    output_format='cif'
)
```

**Important:** Input must be single, ordered, defect-free structure

---

## Phase 5: Perturbation/Augmentation

**Tool:** `pymatgen_perturbation_generator`

**Physical basis:** Random atomic displacements + lattice strain

### Use Case 1: DFT Relaxation Initialization

```python
perturbation_generator(
    input_structures=structure,
    displacement_max=0.05,  # 5% of nearest-neighbor distance
    strain_percent=None,
    n_structures=1,
    seed=42,
    output_format='poscar'
)
```

**Purpose:** Break symmetry to avoid saddle points in DFT relaxation

### Use Case 2: ML Dataset Augmentation

```python
perturbation_generator(
    input_structures=[struct1, struct2, ...],
    displacement_max=0.15,
    strain_percent=[-2, 2],  # ±2% volumetric strain
    n_structures=10,
    seed=42,
    output_format='cif'
)
```

**Purpose:** Generate diverse training data for ML potentials

### Use Case 3: Defect Relaxation

```python
perturbation_generator(
    input_structures=defect_supercell,
    displacement_max=0.08,
    strain_percent=None,
    n_structures=3,
    output_format='poscar'
)
```

**Purpose:** Multiple starting geometries for defect relaxation

### Parameter Guidelines

| Use Case | `displacement_max` | `strain_percent` | `n_structures` |
|----------|-------------------|------------------|----------------|
| DFT initialization | 0.03-0.05 | None | 1 |
| ML augmentation | 0.10-0.20 | [-3, 3] | 5-20 |
| Defect relaxation | 0.05-0.10 | None | 2-5 |
| Thermal sampling | 0.05-0.15 | [-2, 2] | 10-50 |

---

## Storage Best Practices

### ASE Database Storage

```python
# Pymatgen tools now return CIF/POSCAR/JSON.
# Convert the returned structure with the dedicated ASE conversion step
# before passing atoms_dict into ase_store_result.
result = pymatgen_*_generator(
    ...,
    output_format='cif'
)

for i, structure in enumerate(result['structures']):
    ase_store_result(
        db_path='candidates.db',
        atoms_dict=converted_atoms_dict,
        key_value_pairs={
            'candidate_id': f'candidate_{i:04d}',
            'compound': result['formulas'][i],  # NOT 'formula' (reserved)
            'generator': 'substitution',
            'campaign': 'cathode_screen_2026',
            'phase': 'chemical_exploration'
        }
    )
```

### Metadata Attachment

```python
key_value_pairs={
    # Identifiers
    'candidate_id': 'phosphor_093',
    'campaign': 'niobate_phosphor_2026',
    
    # Generation details
    'generator': 'disorder_generator',
    'base_composition': 'SrNb2O6',
    'doping_element': 'Sm',
    'doping_concentration': 0.03,
    
    # Disorder handling flags
    'is_disordered': True,
    'requires_ordering': 'majority',
    
    # Provenance
    'generation_date': '2026-05-08',
    'mp_id_base': 'mp-######'
}
```

### Export for DFT

```python
# For VASP calculations
result = pymatgen_*_generator(
    ...,
    output_format='poscar'
)

for i, poscar_string in enumerate(result['structures']):
    with open(f'POSCAR_{i:04d}', 'w') as f:
        f.write(poscar_string)
```

---

## Connecting Phases

### Sequential Workflow Example

```python
# Phase 0: Composition discovery
compositions = composition_enumerator(elements=['Li', 'V', 'O'], ...)

# Phase 1: Build seed structures
seeds = []
for comp in compositions:
    if comp not in mp_database:
        seed = pymatgen_prototype_builder(spacegroup=225, species=comp['elements'], ...)
        seeds.append(seed)

# Phase 2: Chemical exploration
variants = []
for seed in seeds:
    variant = pymatgen_substitution_generator(
        input_structures=seed,
        substitutions={'Li': ['Na', 'K']},
        ...
    )
    variants.extend(variant['structures'])

# Phase 3: Disorder resolution (if needed)
ordered = []
for structure in variants:
    if not structure.is_ordered:
        result = pymatgen_majority_orderer(input_structures=structure, ...)
        ordered.append(result['structures'][0])
    else:
        ordered.append(structure)

# Phase 4: Storage
for struct in ordered:
    ase_store_result(db_path='final.db', atoms_dict=struct, ...)
```

### Branching Workflow Example

```python
# Start with MP structure
base = mp_get_material_properties(material_ids=['mp-24850'])

# Branch A: Ion exchange for battery analogues
ion_exchanged = pymatgen_ion_exchange_generator(
    input_structures=base['structure'],
    replace_ion='Li+',
    with_ions=['Na+', 'Mg2+', 'Ca2+'],
    ...
)

# Branch B: Substitution for composition space
substituted = pymatgen_substitution_generator(
    input_structures=base['structure'],
    substitutions={'Co': ['Ni', 'Mn', 'Fe']},
    ...
)

# Merge branches
all_candidates = ion_exchanged['structures'] + substituted['structures']

# Continue to disorder/defects as needed
```
