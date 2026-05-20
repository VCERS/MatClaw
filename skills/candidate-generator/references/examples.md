# Common Generation Patterns

Complete examples showing typical candidate generation workflows.

Note: pymatgen tools in these examples return CIF/POSCAR/JSON. When a workflow stores results in an ASE database, include a separate ASE-conversion step before calling `ase_store_result`.

---

## Pattern 1: Isostructural Analogue Screen

**Goal:** Screen all combinations of elements while keeping crystal structure

**Workflow:** Seed structure → Substitution → Storage

```python
# Step 1: Build rock-salt seed structure
seed_result = pymatgen_prototype_builder(
    spacegroup=225,  # Fm-3m (rock-salt)
    species=['Li', 'O'],
    lattice_parameters=[4.33],
    output_format='cif'
)

# Step 2: Systematic element substitution
variants = pymatgen_substitution_generator(
    input_structures=seed_result['structures'][0],
    substitutions={
        'Li': ['Na', 'K', 'Rb'],
        'O': ['S', 'Se', 'Te']
    },
    n_structures=9,  # 3 cations × 3 anions = 9 combinations
    max_attempts=18,  # 2× for safety
    output_format='cif'
)

# Step 3: Store in ASE database
for i, struct_cif in enumerate(variants['structures']):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='rocksalt_analogues.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'rocksalt_{i:03d}',
            'compound': variants['formulas'][i],
            'campaign': 'rocksalt_screen_2026',
            'prototype': 'rocksalt_225'
        }
    )

print(f"Generated {variants['count']} rocksalt analogues")
# Expected output: NaS, NaSe, NaTe, KS, KSe, KTe, RbS, RbSe, RbTe
```

---

## Pattern 2: Battery Cathode Analogue (Li → Na)

**Goal:** Create sodium-ion battery analogues from lithium cathodes

**Workflow:** MP structure → Ion exchange → Storage

```python
# Step 1: Get LiCoO2 structure from Materials Project
licoo2 = mp_get_material_properties(
    material_ids=['mp-24850'],  # LiCoO2
    properties=['structure']
)

# Step 2: Exchange Li → Na with charge neutrality
exchanged = pymatgen_ion_exchange_generator(
    input_structures=licoo2['properties'][0]['structure'],
    replace_ion='Li+',
    with_ions=['Na+'],
    exchange_fraction=1.0,  # Complete replacement
    output_format='cif'
)

# Step 3: Store result
atoms_dict = converted_atoms_dict  # Produced by the dedicated ASE conversion step
ase_store_result(
    db_path='na_cathodes.db',
    atoms_dict=atoms_dict,
    key_value_pairs={
        'compound': 'NaCoO2',
        'base_material': 'LiCoO2',
        'mp_id_base': 'mp-24850',
        'campaign': 'na_ion_cathodes',
        'application': 'battery_cathode'
    }
)

print(f"Created {exchanged['formulas'][0]} from LiCoO2")
```

---

## Pattern 3: High-Entropy Oxide SQS

**Goal:** Model 5-component random alloy (high-entropy material)

**Workflow:** Disorder → SQS ordering → Storage

```python
# Step 1: Get base rocksalt structure
mgo = mp_get_material_properties(material_ids=['mp-1265'])  # MgO

# Step 2: Create disordered 5-component structure
disordered = pymatgen_disorder_generator(
    input_structures=mgo['properties'][0]['structure'],
    site_substitutions={
        'Mg': {
            'Mg': 0.2,
            'Co': 0.2,
            'Ni': 0.2,
            'Cu': 0.2,
            'Zn': 0.2
        }
    },
    output_format='cif'
)

# Step 3: Generate SQS ordered approximants
sqs = pymatgen_sqs_orderer(
    input_structures=disordered['structures'][0],
    supercell_size=20,  # 80-atom supercell
    n_structures=5,  # Generate 5 different SQS variants
    n_mc_steps=500000,  # High for 5 components
    output_format='cif'
)

# Step 4: Store best SQS (lowest sqs_error)
for i, struct_cif in enumerate(sqs['structures'][:3]):  # Top 3
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='high_entropy_oxides.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'heo_sqs_{i+1}',
            'compound': sqs['formulas'][i],
            'campaign': 'high_entropy_oxides',
            'sqs_error': sqs['metadata'][i]['sqs_error'],
            'supercell_atoms': sqs['metadata'][i]['n_sites']
        }
    )

print(f"Generated {len(sqs['structures'])} SQS structures")
print(f"Best SQS error: {sqs['metadata'][0]['sqs_error']:.4f}")
```

---

## Pattern 4: Ground-State Ordering Search

**Goal:** Find energetically favorable ordered configurations

**Workflow:** Disordered → Enumeration → Ewald ranking → Storage

```python
# Step 1: Get disordered Li0.5CoO2 from MP or create
base = mp_get_material_properties(material_ids=['mp-######'])

# Alternatively, create disordered structure
disordered = pymatgen_disorder_generator(
    input_structures=licoo2_full,
    site_substitutions={'Li': {'Li': 0.5, 'vacancy': 0.5}},
    output_format='cif'
)

# Step 2: Enumerate all distinct orderings
ordered = pymatgen_enumeration_orderer(
    input_structures=disordered['structures'][0],
    supercell_size=2,  # 2×2×2 = 8× unit cell
    n_structures=100,  # Get many configurations
    sort_by='ewald',  # Rank by electrostatic energy
    output_format='cif'
)

# Step 3: Store top 10 by Ewald energy
for i, struct_cif in enumerate(ordered['structures'][:10]):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='ground_state_search.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'licoo2_config_{i+1:03d}',
            'compound': ordered['formulas'][i],
            'ewald_energy': ordered['metadata'][i]['ewald_energy'],
            'rank': i + 1,
            'campaign': 'ground_state_li05coo2'
        }
    )

print(f"Top 10 configurations by Ewald energy:")
for i in range(10):
    print(f"  {i+1}. {ordered['formulas'][i]}: "
          f"{ordered['metadata'][i]['ewald_energy']:.3f} eV")
```

---

## Pattern 5: Lanthanide-Doped Phosphor Screen

**Goal:** Screen 93 lanthanide-doped niobate phosphors for luminescence

**Workflow:** MP base → Disorder (dilute doping) → Batch storage

```python
# Step 1: Get base host structure
host_mps = {
    'SrNb2O6': 'mp-4718',
    'BaNb2O6': 'mp-5058',
    'CaNb2O6': 'mp-4863'
}

dopants = ['Sm', 'Eu', 'Tb', 'Dy', 'Er', 'Tm', 'Yb']
concentrations = [0.01, 0.03, 0.05]

all_candidates = []

# Step 2: Generate all combinations
for host_formula, mp_id in host_mps.items():
    # Get host structure
    host = mp_get_material_properties(material_ids=[mp_id])
    
    # Determine doping site (A-site vs B-site)
    if 'Sr' in host_formula:
        doping_site = 'Sr'
    elif 'Ba' in host_formula:
        doping_site = 'Ba'
    elif 'Ca' in host_formula:
        doping_site = 'Ca'
    
    # Generate disordered structures for each dopant/concentration
    for dopant in dopants:
        for conc in concentrations:
            # Normalize for Nb2 (divide by 2) if doping B-site
            if doping_site == 'Nb':
                conc_normalized = conc / 2
            else:
                conc_normalized = conc
            
            result = pymatgen_disorder_generator(
                input_structures=host['properties'][0]['structure'],
                site_substitutions={
                    doping_site: {
                        dopant: conc_normalized,
                        doping_site: 1.0 - conc_normalized
                    }
                },
                output_format='cif'
            )
            
            # Attach metadata
            candidate = {
                'structure': result['structures'][0],
                'formula': result['formulas'][0],
                'metadata': {
                    'is_disordered': True,
                    'doping_type': 'dilute',
                    'requires_ordering': 'majority',
                    'doping_concentration': conc,
                    'host_formula': host_formula,
                    'dopant_species': [dopant],
                    'mp_id_host': mp_id
                }
            }
            all_candidates.append(candidate)

# Step 3: Store all candidates
for i, candidate in enumerate(all_candidates):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='lanthanide_phosphors.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'phosphor_{i:03d}',
            'compound': candidate['formula'],
            'host': candidate['metadata']['host_formula'],
            'dopant': candidate['metadata']['dopant_species'][0],
            'doping_conc': candidate['metadata']['doping_concentration'],
            'is_disordered': True,
            'requires_ordering': 'majority',
            'campaign': 'lanthanide_phosphor_screen_2026'
        }
    )

print(f"Generated {len(all_candidates)} phosphor candidates")
# Expected: 3 hosts × 7 dopants × 3 concentrations = 63 structures
```

---

## Pattern 6: Oxygen Vacancy Defects

**Goal:** Generate vacancy defects for oxygen conductor screening

**Workflow:** Base structure → Defect generation → Perturbation → Storage

```python
# Step 1: Get perfect host structure
ceo2 = mp_get_material_properties(material_ids=['mp-20194'])  # CeO2

# Step 2: Generate oxygen vacancy supercells
defects = pymatgen_defect_generator(
    input_structure=ceo2['properties'][0]['structure'],
    vacancy_species=['O'],
    supercell_min_atoms=96,
    output_format='cif'
)

# Step 3: Perturb for DFT relaxation initialization
all_structures = []
for defect_struct in defects['structures']:
    perturbed = pymatgen_perturbation_generator(
        input_structures=defect_struct,
        displacement_max=0.05,
        n_structures=3,  # 3 initial geometries per defect
        seed=42,
        output_format='cif'
    )
    all_structures.extend(perturbed['structures'])

# Step 4: Store all perturbed defect structures
for i, struct_cif in enumerate(all_structures):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='oxygen_vacancies.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'vacancy_{i:03d}',
            'defect_type': 'oxygen_vacancy',
            'perturbed': True,
            'campaign': 'oxygen_conductor_vacancies'
        }
    )

print(f"Generated {len(defects['structures'])} unique vacancy sites")
print(f"Total structures with perturbations: {len(all_structures)}")
```

---

## Pattern 7: Perovskite B-Site Doping Screen

**Goal:** Screen dozens of B-site dopants in perovskite oxides

**Workflow:** Base structure → Substitution → Optional disorder → Storage

```python
# Step 1: Get BaTiO3 perovskite base
batio3 = mp_get_material_properties(material_ids=['mp-2998'])

# Step 2: Screen B-site metals
b_site_metals = ['Ti', 'Zr', 'Hf', 'Sn', 'Ce', 'Nb', 'Ta', 'Mo', 'W']

substituted = pymatgen_substitution_generator(
    input_structures=batio3['properties'][0]['structure'],
    substitutions={'Ti': b_site_metals},
    n_structures=len(b_site_metals),
    max_attempts=len(b_site_metals) * 2,
    output_format='cif'
)

# Step 3: Optional - Create mixed B-site (disorder)
# For Ba(Ti0.8Zr0.2)O3 type materials
mixed_b_site = []
for base_metal in ['Ti', 'Zr']:
    for dopant in ['Nb', 'Ta']:
        disordered = pymatgen_disorder_generator(
            input_structures=batio3['properties'][0]['structure'],
            site_substitutions={
                base_metal: {
                    base_metal: 0.9,
                    dopant: 0.1
                }
            },
            output_format='cif'
        )
        mixed_b_site.append(disordered['structures'][0])

# Step 4: Store all structures
for i, struct_cif in enumerate(substituted['structures']):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='perovskite_bsite_scan.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'perovskite_pure_{i:03d}',
            'compound': substituted['formulas'][i],
            'substitution_type': 'pure_bsite',
            'campaign': 'perovskite_screen_2026'
        }
    )

for i, struct_cif in enumerate(mixed_b_site):
    atoms_dict = converted_mixed_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='perovskite_bsite_scan.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'perovskite_mixed_{i:03d}',
            'substitution_type': 'mixed_bsite_10pct',
            'is_disordered': True,
            'requires_ordering': 'majority'
        }
    )

print(f"Pure B-site substitutions: {substituted['count']}")
print(f"Mixed B-site structures: {len(mixed_b_site)}")
```

---

## Pattern 8: ML Training Set Generation

**Goal:** Generate diverse structures for ML potential training

**Workflow:** Base structures → Perturbation (augmentation) → Storage

```python
# Step 1: Get base structures from MP
base_formulas = ['LiFePO4', 'LiCoO2', 'LiMn2O4', 'Li2FeSiO4']
base_structures = []

for formula in base_formulas:
    mp_result = mp_search_materials(formula=formula, limit=1)
    if mp_result['count'] > 0:
        props = mp_get_material_properties(
            material_ids=[mp_result['material_ids'][0]]
        )
        base_structures.append(props['properties'][0]['structure'])

# Step 2: Generate augmented training set
training_set = []

for base_struct in base_structures:
    # Generate 20 perturbed structures per base
    perturbed = pymatgen_perturbation_generator(
        input_structures=base_struct,
        displacement_max=0.15,  # 15% displacement for diversity
        strain_percent=[-3, 3],  # ±3% volumetric strain
        n_structures=20,
        seed=42,
        output_format='cif'
    )
    training_set.extend(perturbed['structures'])

# Step 3: Store training set
for i, struct_cif in enumerate(training_set):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='ml_training_set.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'dataset_id': f'train_{i:04d}',
            'augmented': True,
            'campaign': 'ml_training_cathodes_2026',
            'purpose': 'training'
        }
    )

print(f"Generated {len(training_set)} training structures")
print(f"From {len(base_structures)} base materials")
# Expected: 4 bases × 20 augmented = 80 structures
```

---

## Pattern 9: Complete Pipeline (All Phases)

**Goal:** Full workflow from a seed material to stored candidates

**Workflow:** Seed formula → MP seed structure → Substitution → Disorder → Storage

```python
# Phase 0: Provide a seed material formula
seed_formula = 'LiNiO2'

# Phase 1: Retrieve the seed structure from Materials Project
seed_search = mp_search_materials(formula=seed_formula, limit=1)
if seed_search['count'] == 0:
    raise ValueError(f"No Materials Project structure found for {seed_formula}")

seed_props = mp_get_material_properties(
    material_ids=[seed_search['material_ids'][0]]
)
seed_structure = seed_props['properties'][0]['structure']

all_candidates = []

# Phase 2: Chemical exploration (keep Li seed + make Na analogue)
variants = pymatgen_substitution_generator(
    input_structures=seed_structure,
    substitutions={'Li': ['Li', 'Na']},  # Keep original + Na variant
    n_structures=2,
    max_attempts=4,
    output_format='cif'
)

# Phase 3: Add disorder (5% Co doping)
for variant_cif in variants['structures']:
    disordered = pymatgen_disorder_generator(
        input_structures=variant_cif,
        site_substitutions={'Ni': {'Ni': 0.95, 'Co': 0.05}},
        output_format='cif'
    )
    
    all_candidates.append({
        'structure': disordered['structures'][0],
        'formula': disordered['formulas'][0],
        'metadata': {
            'seed_formula': seed_formula,
            'seed_mp_id': seed_search['material_ids'][0],
            'is_disordered': True,
            'requires_ordering': 'majority'
        }
    })

# Storage
for i, candidate in enumerate(all_candidates):
    atoms_dict = converted_atoms_dicts[i]  # Produced by the dedicated ASE conversion step
    ase_store_result(
        db_path='complete_pipeline.db',
        atoms_dict=atoms_dict,
        key_value_pairs={
            'candidate_id': f'pipeline_{i:03d}',
            'compound': candidate['formula'],
            **candidate['metadata']
        }
    )

print(f"Generated {len(all_candidates)} candidates through full pipeline")
```

---

## Connecting to Screening

After generation, candidates go to `candidate-screener` skill:

```python
# Query ASE database
candidates = ase_query(
    db_path='candidates.db',
    property_filters={'campaign': 'cathode_screen_2026'}
)

# Pass to candidate-screener
# The screener will:
# 1. Validate structures
# 2. Retrieve properties (MP → ASE → ML hierarchy)
# 3. Apply screening criteria
# 4. Rank by multi-objective optimization
```
