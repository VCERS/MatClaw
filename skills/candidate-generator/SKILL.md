---
name: candidate-generator
description: |
  Generate inorganic crystal structure candidates for computational materials discovery workflows.
  
  **TRIGGER THIS SKILL when user mentions:**
  - "generate candidates", "create structures", "build structures", "structure generation"
  - "screen materials", "explore compositions", "chemical substitution", "doping"
  - "isostructural analogues", "battery cathodes", "perovskites", "solid solutions"
  - "enumerate configurations", "SQS generation", "disorder", "defects"
  - "high-throughput", "DFT screening", "ML training set", "candidate pool"
  - Element lists like "Li-Mn-P-O system", "transition metal oxides"
  - Number requests: "generate 50 structures", "100 candidates"
  
  **Covers COMPLETE pipeline:**
  Elements-only entry → Composition discovery → Seed structures → Chemical space exploration → 
  Disorder/ordering → Defect generation → Perturbation → ASE database storage
    
  **Detailed references available in references/ directory**
---

# Inorganic Candidate Generation Skill

## Core Philosophy

Candidate generation is a **funnel process**: start broad (many compositions, chemistries, configurations), 
then narrow using physical filters (charge neutrality, Ewald energy, thermodynamic stability). 
The workflow is modular and nonlinear—skip phases that don't apply to your discovery goal.

**Workflow phases:**
```
Elements → Compositions → Seed Structures → Chemical Variants → Order/Disorder → Defects → Perturbation → Storage
```

**Entry points:**
- **Elements only** (Li-Mn-P-O) → Phase 0 (composition discovery)
- **Composition known** (LiMnPO₄) → Phase 1 (seed structure) or Phase 2 (if MP structure exists)
- **Structure exists** (from MP/CIF/ASE) → Phase 2 (chemical exploration) or later phases

**Critical rules:**
1. **Always use MCP tools** — Never write custom generators or formula-only scripts
2. **Store in ASE database** — Use `ase_store_result` with `output_format='ase'`
3. **Real structures required** — All outputs must have atomic positions, lattice, spacegroup
4. **Plan for large-scale** — If N > 20, create planning file first (see [references/large-scale-planning.md](references/large-scale-planning.md))

> **Why MCP tools matter:**
> - Generate real crystal structures (CIF/POSCAR) ready for DFT/ML
> - Provide thermodynamic validation (stability, energy above hull)
> - Compute structural properties (spacegroup, coordination, bonds)
> - Custom scripts produce formula strings without structures = scientifically invalid

---

## Quick Tool Reference

**For complete tool specifications with all parameters, see [references/tool-catalog.md](references/tool-catalog.md)**

### By Workflow Phase

| Phase | Tool | Purpose | Key Parameters |
|-------|------|---------|----------------|
| **Phase 0: Composition Discovery** | | | |
| | `composition_enumerator` | Generate charge-balanced compositions | `elements`, `oxidation_states`, `max_formula_units` |
| | `pymatgen_substitution_predictor` | ICSD-based element substitution | `composition`, `threshold` |
| | `mp_search_materials` | Find MP template structures | `elements`, `is_stable` |
| **Phase 1: Seed Structure** | | | |
| | `pymatgen_prototype_builder` | Build from spacegroup | `spacegroup`, `species`, `lattice_parameters` |
| **Phase 2: Chemical Exploration** | | | |
| | `pymatgen_substitution_generator` | **Ordered enumeration** (integer occupancy) | `substitutions`, `n_structures`, `max_attempts` |
| | `pymatgen_ion_exchange_generator` | Charge-neutral ion substitution | `replace_ion`, `with_ions`, `exchange_fraction` |
| **Phase 3: Disorder** | | | |
| | `pymatgen_disorder_generator` | **Fractional occupancy** (statistical disorder) | `site_substitutions` |
| | `pymatgen_enumeration_generator` | Exhaustive ordered configurations | `supercell_size`, `sort_by='ewald'` |
| | `pymatgen_sqs_generator` | Special quasirandom structures | `supercell_size`, `n_mc_steps` |
| **Phase 4: Defects** | | | |
| | `pymatgen_defect_generator` | Point defect supercells | `vacancy_species`, `substitution_species`, `interstitial_species` |
| **Phase 5: Perturbation** | | | |
| | `pymatgen_perturbation_generator` | Rattle atoms + strain lattice | `displacement_max`, `strain_percent` |

### Critical Tool Distinction

**`disorder_generator` vs `substitution_generator`:**

| Aspect | `disorder_generator` | `substitution_generator` |
|--------|---------------------|-------------------------|
| **Output** | Fractional occupancy | Integer occupancy (ordered) |
| **Site occupancy** | 80% Ni + 20% Mn on same site | Site 1: 100% Mn; Sites 2-5: 100% Ni |
| **Example** | Li₃[Ni₂.₄Mn₀.₆]O₆ (statistical) | LiNi₄MnO₁₀ (ordered variant) |
| **Output count** | 1 disordered structure | Multiple ordered configurations |
| **Use for** | SQS generation, VCA | Supercell enumeration, DFT screening |

**Rule:** For partial substitution like Li[Ni₀.₈Mn₀.₂]O₂:
- Want fractional occupancy (every site 80%Ni+20%Mn)? → `disorder_generator`
- Want ordered enumeration (1 specific Ni replaced)? → `substitution_generator`

---

## Workflow Phases Overview

### Phase 0: Composition Discovery (CONDITIONAL)

**When to use:** You only know elements (e.g., "Li-Mn-P-O"), not specific compositions

**Skip if:** You already have target composition or structure

**Three strategies:**
1. **Exhaustive enumeration** — Use `composition_enumerator` for systematic exploration
2. **Template-based** — Use `mp_search_materials` to find analogues, extract patterns
3. **ICSD substitution** — Use `pymatgen_substitution_predictor` from known material

**Decision tree:**
- Known analogue exists? → Template-based + ICSD substitution
- Well-studied system? → Template-based first, enumeration if gaps
- Exploratory discovery? → Exhaustive enumeration → filter by stability

**Output:** Ranked list of stable/metastable compositions

**Next:** For each composition, check MP for structures. If found → Phase 2; if not → Phase 1

**Detailed guidance:** See [references/composition-discovery.md](references/composition-discovery.md)

---

### Phase 1: Seed Structure (CONDITIONAL)

**When to use:** Need to build structure from scratch (no MP structure available)

**Skip if:** Structure already exists from MP, CIF, or ASE database

**Tool:** `pymatgen_prototype_builder`

**Common prototypes:**
- Rock-salt (225, Fm-3m): NaCl, LiF, MgO
- Perovskite (221, Pm-3m): BaTiO₃, SrTiO₃
- Spinel (227, Fd-3m): MgAl₂O₄, LiMn₂O₄
- Layered oxide (166, R-3m): LiCoO₂, LiNiO₂
- Olivine (62, Pnma): LiFePO₄, LiMnPO₄

**Example:**
```python
seed = pymatgen_prototype_builder(
    spacegroup=225,  # Rock-salt
    species=['Li', 'O'],
    lattice_parameters=[4.33]  # cubic
)
```

**Next:** Phase 2 (chemical exploration)

---

### Phase 2: Chemical Space Exploration (CONDITIONAL)

**When to use:** Want to explore different compositions/dopings while keeping structure

**Skip if:** Want to keep exact composition

**Branch A — Charge-neutral (ionic materials):**
- Tool: `pymatgen_ion_exchange_generator`
- Use for: Battery materials, ionic conductors, charge-balanced doping
- Automatically adjusts stoichiometry for charge neutrality

**Branch B — Exploratory (screening):**
- Tool: `pymatgen_substitution_generator`
- Use for: Isostructural analogues, ML training sets, exploratory screening
- Generates ordered structures with integer occupancy

**Decision:**
- Material is ionic + charge balance critical? → Branch A (ion_exchange)
- Exploratory screening / charge handled post-hoc? → Branch B (substitution)

**Examples:**
```python
# Branch A: Li → Na battery cathode analogue
ion_exchange_generator(
    replace_ion='Li',
    with_ions=['Na'],
    exchange_fraction=1.0
)

# Branch B: Screen B-site metals in perovskite
substitution_generator(
    substitutions={'Ti': ['Zr', 'Hf', 'Sn']},
    n_structures=1,
    enforce_charge_neutrality=False
)
```

**Next:** Phase 3 (if structures have disorder) or Phase 4 (defects) or Phase 5 (perturbation)

---

### Phase 3: Disorder Resolution (CONDITIONAL)

**When to use:** Structures have fractional site occupancies

**Skip if:** All structures fully ordered

**Creating disorder (order → disorder):**
- Tool: `pymatgen_disorder_generator`
- Use for: Li[Ni₀.₈Mn₀.₂]O₂-type fractional substitutions
- Creates statistical disorder (all sites get fractional occupancy)

**Resolving disorder (disorder → ordered):**

The choice between disorder approaches depends critically on **doping concentration** when generating structures for screening workflows:

**Branch A — Low doping screening (< 10% dopant concentration):**
- Tool: `pymatgen_disorder_generator` → disordered unit cell
- Use for: Fast high-throughput screening with small unit cells
- Creates fractional occupancy (e.g., Sr₀.₉₇Sm₀.₀₃MoO₄)
- **Physical basis:** In the dilute limit, the host lattice structure and properties dominate. Dopants provide minor perturbations but don't fundamentally alter bonding or electronic structure. The majority-species approximation (keeping only the dominant species per site) is physically justified.
- **Downstream handling:** Screening tools like `matgl_relax_structure` automatically apply majority-species approximation at the tool level (no additional steps needed)
- **When valid:** Doping concentrations where dopant-dopant interactions are negligible (typically < 10%)

**Branch B — High doping / solid solution modeling (> 20% dopant concentration):**
- Tool: `pymatgen_sqs_generator` → ordered supercell (50-200 atoms)
- Use for: Model random alloys, solid solutions, high-entropy materials where disorder is functionally important
- Returns fully ordered quasirandom approximant
- **Physical basis:** At high concentrations, dopant-dopant interactions and local ordering matter. The spatial arrangement of dopants affects properties (electronic structure, phonons, stability). SQS structures capture the statistical correlation functions of true random disorder while remaining computationally tractable.
- **Downstream handling:** Already fully ordered → screening tools like `matgl_relax_structure` work directly (no approximations)
- **Computational cost:** Larger supercells (10-50× more atoms) but more accurate for concentrated systems
- Increase `n_mc_steps` for multicomponent systems (50k → 100k-500k for ternary/quaternary)

**Branch C — Intermediate concentration (10-20% dopant):**
- Strategy: Initial screening with disorder_generator (fast), then validate top candidates with sqs_generator (accurate)
- Rationale: Screen cheaply across many candidates, invest in accurate modeling only for promising materials

**Decision tree for screening workflows:**
```
Doping concentration known?
├─ < 10%: disorder_generator (unit cell, fast)
├─ 10-20%: disorder_generator (screen) → sqs_generator (validate top 10)
└─ > 20%: sqs_generator (supercell, accurate)

High-entropy (≥4 mixing species)?: sqs_generator (enumeration intractable)
```

**Next:** Phase 4 (defects) or Phase 5 (perturbation) or storage

---

### Phase 4: Defect Generation (OPTIONAL)

**When to use:** Need point defect supercells (vacancies, substitutions, interstitials)

**Skip if:** Only need perfect bulk structures

**Tool:** `pymatgen_defect_generator`

**Important:** Pass single, ordered, defect-free host structure (not multiple structures)

**Example:**
```python
defect_generator(
    input_structure=perfect_host,  # Single structure only!
    vacancy_species=['Li'],
    substitution_species={'Mn': ['Fe', 'Co']},
    supercell_min_atoms=64
)
```

**Outputs:** One supercell per symmetry-inequivalent defect site

**Next:** Phase 5 (perturbation recommended for defects) or storage

---

### Phase 5: Perturbation/Augmentation (OPTIONAL)

**When to use:**
- Break symmetry before DFT (avoid saddle points)
- ML dataset augmentation
- Probe elastic/thermal response

**Tool:** `pymatgen_perturbation_generator`

**Parameters by use case:**
- **DFT relaxation:** `displacement_max=0.05`, `strain_percent=None`, `n_structures=1`
- **ML augmentation:** `displacement_max=0.15`, `strain_percent=[-2, 2]`, `n_structures=10`
- **Defect relaxation:** `displacement_max=0.08`, `strain_percent=None`, `n_structures=3`

**Next:** Storage in ASE database

---

## Storage and Validation

### Store in ASE Database

**Critical:** Always use `output_format='ase'` when feeding to `ase_store_result`

```python
# Generate with ASE format
result = pymatgen_substitution_generator(
    input_structures=structure,
    substitutions={'Li': 'Na'},
    output_format='ase'  # REQUIRED for ASE database
)

# Store each structure
for s in result['structures']:
    ase_store_result(
        db_path='candidates.db',
        atoms_dict=s['structure'],
        key_value_pairs={
            'compound': s['formula'],  # NOT 'formula' (reserved)
            'generator': 'substitution',
            'campaign': 'cathode_screen_2026'
        }
    )
```

**ASE reserved keys to AVOID:**
`id`, `unique_id`, `formula`, `spacegroup`, `energy`, `forces`, `cell`, `natoms`, etc.

**Use instead:** `compound`, `sg_num`, `candidate_id`, etc.

### Optional MP Stability Check

```python
# Filter by thermodynamic stability
for structure in final_structures:
    mp_result = mp_search_materials(formula=structure['formula'])
    
    if mp_result['count'] > 0:
        # Composition exists in MP and likely stable
        structure['mp_stable'] = True
    else:
        # Novel or metastable
        structure['requires_dft'] = True
```

---

## Decision Algorithm

**For complete decision trees and parameter calculation rules, see [references/decision-trees.md](references/decision-trees.md)**

### Quick Workflow Decision

```
1. Have existing structure? 
   → YES: use it | NO: pymatgen_prototype_builder
   
2. Want new chemistries?
   → YES: Ionic + charge critical? 
      → YES: ion_exchange_generator
      → NO: substitution_generator
   
3. Structures have partial occupancies?
   → NO: skip | YES: Need ALL orderings?
      → YES: enumeration_generator (supercell_size ≤ 2)
      → NO: Modeling disorder? 
         → YES: sqs_generator
         
4. Need defects?
   → YES: defect_generator (single structure)
   
5. Need perturbations?
   → YES: perturbation_generator
   
6. Store with ase_store_result (output_format='ase')
```

### Large-Scale Generation (>20 structures)

**If user requests >20 structures:**
1. **Create planning file FIRST** (don't execute immediately)
2. Organize into scientific batches
3. Present plan to user for approval
4. **Create project-specific Python script** using examples/batch_generation_example.py as reference
5. User executes the script to generate structures
6. Export final results

> **⚠️ CRITICAL: DO NOT EDIT THE EXAMPLE SCRIPT DIRECTLY**
> 
> The file `examples/batch_generation_example.py` is a **REFERENCE TEMPLATE** only.
> **NEVER** edit it directly for project-specific use.
> 
> **Instead:** Read the example script, then **CREATE A NEW** project-specific script in the user's
> working directory (e.g., `matclaw-tests/.../batch_generation.py` or `run_generation.py`).
> Adapt the logic as needed for the specific project structure and requirements.

**Template Features:**
- ✅ **Dynamic tool selection** - Reads tool name from plan (not hardcoded)
- ✅ **Flexible base structure resolution** - 5 fallback options
- ✅ **Generic parameter handling** - Works with any tool
- ✅ **Automatic checkpoint/resume** - Progress tracked in plan file
- ✅ **Comprehensive customization guide** - Clear adaptation instructions

**Workflow:**
- Agent creates `generation_candidates.json` with all candidate specifications
- Agent creates **NEW** project-specific Python script based on the example template
- User runs the project-specific script:
  ```bash
  python run_generation.py  # or project-specific script name
  ```
- Script handles: checkpointing, resume, progress tracking, error handling

> **⚠️ Output Format Compatibility:**
> 
> Ensure `output_format` compatibility between planning file and batch script:
> - If script saves to CIF files: Force `tool_params['output_format'] = 'cif'` (override plan)
> - If script saves to ASE database: Force `tool_params['output_format'] = 'ase'` (override plan)
> - If script expects VASP: Force `tool_params['output_format'] = 'poscar'` (override plan)
> 
> **Don't rely on plan parameters** — explicitly set output_format in the script to match
> downstream file/database requirements. Tools return different types:
> - `'cif'`/`'poscar'` → strings
> - `'ase'` → dictionaries with `{numbers, positions, cell, pbc}`

**See:** [references/large-scale-planning.md](references/large-scale-planning.md) for complete planning workflow

---

## Common Patterns

**Brief examples showing typical workflows**

### Isostructural Analogue Screen

```python
# 1. Build rock-salt seed
seed = pymatgen_prototype_builder(
    spacegroup=225, 
    species=['Li','O'], 
    lattice_parameters=[4.33]
)

# 2. Swap elements: Li → Na,K,Rb; O → S,Se
variants = pymatgen_substitution_generator(
    input_structures=seed['structures'][0],
    substitutions={'Li': ['Na', 'K', 'Rb'], 'O': ['S', 'Se']},
    n_structures=1,  # Deterministic swaps
    max_attempts=6,
    output_format='ase'
)

# 3. Store in ASE database
for s in variants['structures']:
    ase_store_result(
        db_path='screen.db',
        atoms_dict=s,
        key_value_pairs={'compound': s['formula'], 'campaign': 'rocksalt'}
    )
```

### Battery Cathode Analogue (Li → Na)

```python
# Get LiCoO2 from MP
licoo2 = mp_get_material_properties(material_ids=['mp-24850'])

# Exchange Li → Na with charge neutrality
exchanged = pymatgen_ion_exchange_generator(
    input_structures=licoo2['properties'][0]['structure'],
    replace_ion='Li',
    with_ions=['Na'],
    exchange_fraction=1.0,
    output_format='ase'
)
```

### High-Entropy Oxide SQS

```python
# Starting from disordered structure with 5-component cation mixing
# Input has fractional occupancies: {Mg:0.2, Co:0.2, Ni:0.2, Cu:0.2, Zn:0.2}
sqs = pymatgen_sqs_generator(
    input_structures=disordered_structure,
    supercell_size=20,
    n_structures=5,
    n_mc_steps=500000,  # High for 5 components
    output_format='ase'
)
# Best SQS is sqs['structures'][0] (sorted by sqs_error)
```

### Ground-State Ordering Search

```python
# Li₀.₅CoO₂ with partial Li occupancy
ordered = pymatgen_enumeration_generator(
    input_structures=disordered_licoo2,
    supercell_size=2,
    n_structures=100,
    sort_by='ewald',
    output_format='ase'
)
# Top 10 by Ewald energy are most plausible ground states
```

---

## Common Pitfalls

**For complete troubleshooting guide, see [references/gotchas.md](references/gotchas.md)**

### Critical Errors to Avoid

1. **Using wrong output_format for ASE** → Always `output_format='ase'` for `ase_store_result`
2. **Using ASE reserved keys** → Never use `formula`, `spacegroup`, `id`, `energy`, etc. in metadata
3. **`substitution_generator` hangs** → Set `max_attempts = n_structures × num_combinations`
4. **`enumeration_generator` hangs** → Keep `supercell_size ≤ 2` for ternary+ systems
5. **Expecting fractional occupancy from `substitution_generator`** → Use `disorder_generator` instead
6. **`ion_exchange_generator` returns 0** → Try different `exchange_fraction` values
7. **Batch script output_format mismatch** → When generating planning file with `output_format='ase'` but batch script expects CIF strings, force `tool_params['output_format'] = 'cif'` in script (override plan parameter)
8. **`disorder_generator` fractional occupancy errors** → Fractions must sum to 1.0 **per site**, not per formula unit. For Bi₂, Nb₂, Y₃ compounds, divide by stoichiometric coefficient (see below)
9. **Materials Project API rate limits** → Large-scale generation (>20 structures) may hit rate limits causing transient failures. Use retry logic (see below)

### Quick Debugging

| Symptom | Cause | Solution |
|---------|-------|----------|
| "Missing required keys: ['numbers']" | Wrong output_format | Set `output_format='ase'` |
| "Bad key" error | ASE reserved name | Use `compound` not `formula` |
| Tool hangs | supercell_size too large | Reduce to 1-2 or switch to SQS |
| count: 0 in result | max_attempts too low | Calculate explicitly |
| High sqs_error | Poor convergence | Increase n_mc_steps |
| "Expected CIF string, got dict" | output_format='ase' in plan but script needs CIF | Force `tool_params['output_format'] = 'cif'` |
| "Fractions sum to 2.0/3.0" | Per-formula instead of per-site | Divide by stoichiometric coefficient |
| Transient "Could not resolve base" | Materials Project rate limit | Add retry logic with exponential backoff |

---

## Large-Scale Generation: Best Practices

### Materials Project API Rate Limits

**Problem:** When generating >20 structures, Materials Project API may hit rate limits causing transient failures.
Structures that should succeed may fail with "Could not resolve base structure" errors.

**Observed behavior:**
- Initial batch run: ~20-30% success rate
- Retry (same code, same parameters): Additional ~30-40% succeed
- Multiple retries may be needed to reach maximum achievable success rate

**Solutions:**

1. **Built-in retry logic in batch scripts:**
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_with_retry(tool_func, **params):
    """Retry tool calls with exponential backoff"""
    return tool_func(**params)

# Usage in batch script
try:
    result = generate_with_retry(pymatgen_disorder_generator, 
                                 input_structures=base, 
                                 site_substitutions=substitutions)
except Exception as e:
    # Log failure and continue
    logging.error(f"Failed after retries: {e}")
```

2. **Manual retry workflow:**
```bash
# First run
python batch_generation.py

# Check results
python generation_summary.py  # Shows completed vs failed

# Reset failed candidates with available base structures
python -c "import json; plan = json.load(open('plan.json')); 
  <reset logic>; json.dump(plan, open('plan.json', 'w'))"

# Retry
python batch_generation.py
```

3. **Rate limit friendly practices:**
- Add `time.sleep(0.5)` between MP API calls
- Use MP structure caching (save fetched structures locally)
- Verify base structure availability before large-scale generation:
  ```python
  # Pre-flight check
  for candidate in candidates:
      result = mp_search_materials(formula=candidate['base_composition'])
      if result['count'] == 0:
          candidate['skip'] = True  # Mark as impossible
  ```

### Fractional Occupancy Normalization

**Problem:** `pymatgen_disorder_generator` requires `site_substitutions` fractions to sum to **1.0 per site**,
not per formula unit. For compounds with multiple atoms per site (Bi₂, Nb₂, Y₃), this requires normalization.

**Examples:**

**❌ WRONG (will fail with "Fractions sum to 2.0/3.0" error):**
```python
# BaBi₂(MoO₄)₄ with 3% Sm doping
# Bi has stoichiometry 2 → naive approach gives 0.03 + 1.97 = 2.0
site_substitutions = {'Bi': {'Sm': 0.03, 'Bi': 1.97}}  # ❌ ERROR

# Y₃Al₅O₁₂ with 3% Sm doping  
# Y has stoichiometry 3 → naive approach gives 0.03 + 2.97 = 3.0
site_substitutions = {'Y': {'Sm': 0.03, 'Y': 2.97}}  # ❌ ERROR
```

**✅ CORRECT (normalize by stoichiometric coefficient):**
```python
# BaBi₂(MoO₄)₄ with 3% Sm doping
# Divide by 2 (Bi stoichiometry): 0.03/2 = 0.015, 1.97/2 = 0.985
site_substitutions = {'Bi': {'Sm': 0.015, 'Bi': 0.985}}  # ✅ Sums to 1.0

# Y₃Al₅O₁₂ with 3% Sm doping
# Divide by 3 (Y stoichiometry): 0.03/3 = 0.01, 2.97/3 = 0.99
site_substitutions = {'Y': {'Sm': 0.01, 'Y': 0.99}}  # ✅ Sums to 1.0

# SrNb₂O₆ with 3% Sm doping
# Divide by 2 (Nb stoichiometry)
site_substitutions = {'Nb': {'Sm': 0.015, 'Nb': 0.985}}  # ✅ Sums to 1.0
```

**Rule:** If element has stoichiometry `n`, divide all occupancy fractions by `n`.

**Validation script:**
```python
def normalize_site_substitutions(formula, site_subs):
    """Normalize site_substitutions to per-site basis"""
    from pymatgen.core import Composition
    comp = Composition(formula)
    
    normalized = {}
    for element, occupancies in site_subs.items():
        stoich = comp[element]  # Get stoichiometric coefficient
        normalized[element] = {
            species: frac / stoich 
            for species, frac in occupancies.items()
        }
        
        # Verify sum = 1.0
        total = sum(normalized[element].values())
        assert abs(total - 1.0) < 0.01, f"Fractions sum to {total}, not 1.0"
    
    return normalized

# Usage
site_subs = normalize_site_substitutions(
    formula='BaBi2(MoO4)4',
    site_subs={'Bi': {'Sm': 0.03, 'Bi': 1.97}}
)
# Returns: {'Bi': {'Sm': 0.015, 'Bi': 0.985}}
```

---

## Connecting to Downstream Workflows

### To candidate-screener Skill

After generating candidates and storing in ASE database:

```python
# Query ASE database for all candidates
candidates = ase_query_db(
    db_path='candidates.db',
    property_filters={'campaign': 'cathode_screen_2026'}
)

# Pass to candidate-screener for property enrichment, filtering, ranking
# The candidate-screener will:
# 1. Validate structures
# 2. Retrieve properties (MP → ASE → ML hierarchy)
# 3. Apply screening criteria
# 4. Rank by multi-objective optimization
```

### To VASP/DFT Calculations

```python
# Generate candidates with POSCAR format for VASP
result = pymatgen_substitution_generator(
    input_structures=structure,
    substitutions={'Li': 'Na'},
    output_format='poscar'  # For VASP
)

# Each structure can be written directly to POSCAR file
for i, s in enumerate(result['structures']):
    with open(f'POSCAR_{i}', 'w') as f:
        f.write(s['structure'])
```

---

## Reference Files

Complete detailed documentation available in `references/` directory:

1. **[tool-catalog.md](references/tool-catalog.md)** — Complete tool specifications with all parameters, returns, examples
2. **[decision-trees.md](references/decision-trees.md)** — Detailed decision logic and parameter calculation rules
3. **[composition-discovery.md](references/composition-discovery.md)** — Complete Phase 0 strategies and examples  
4. **[gotchas.md](references/gotchas.md)** — Troubleshooting guide with common errors and solutions
5. **[large-scale-planning.md](references/large-scale-planning.md)** — Planning workflow for >20 structures with checkpointing

---

## Summary

**This skill provides guidance on:**
- **WHAT** tool to use for each generation scenario
- **WHY** certain approaches are appropriate
- **HOW** to connect tools into multi-phase workflows
- **WHEN** to use planning for large-scale generation

**Key principles:**
1. Always use MCP tools (never write custom generators)
2. Start broad, narrow with physical filters
3. Store everything in ASE database
4. Use `output_format='ase'` for ASE storage
5. Plan first for >20 structures

**Entry point decision:**
- Elements only? → Phase 0 (composition discovery)
- Composition known? → Phase 1 (seed) or Phase 2 (if MP exists)
- Structure exists? → Phase 2+ (exploration, disorder, defects, perturbation)

**For complete details, algorithms, and troubleshooting:** See reference files in `references/` directory.
