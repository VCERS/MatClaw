---
name: candidate-generator
description: |
  **Guide materials discovery workflows from concept to crystal structures.** 
  
  Generate, enumerate, and organize inorganic crystal structure candidates for screening, DFT, ML training, and experimental synthesis planning. Handles complete pipeline from elements-only input to ASE database storage with provenance tracking.
  
  **Trigger this skill for:**
  - Structure generation: "generate candidates", "create structures", "build crystal structures", "make supercells"
  - Chemical exploration: "screen compositions", "chemical substitution", "doping", "solid solutions", "ion exchange"  
  - Materials classes: "battery cathodes", "perovskites", "isostructural analogues", "high-entropy oxides", "defects"
  - Configuration space: "enumerate orderings", "SQS generation", "disorder-to-ordered", "vacancy defects", "interstitials"
  - Scale: "generate 50 structures", "100 candidates", "high-throughput", "ML training set", "candidate library"
  - Starting points: Element lists ("Li-Mn-P-O system"), formulas ("LiCoO2 analogues"), structure types ("olivine"), ICSD/MP/COD IDs
  - Workflows: "prototype matching", "lattice perturbation", "supercell expansion"
  
  **Complete pipeline coverage:**
  Elements → Compositions → Seed structures → Chemical exploration → Disorder/ordering → Defects → Perturbation → ASE database
  
  **When NOT to trigger:** Reading/parsing existing structures (just use pymatgen directly), analyzing calculated properties (use candidate-screener), validating geometry (use structure_validator).
  
    **Bundled resources:** Phase guide + worked examples → references/ directory
---

# Inorganic Candidate Generation Skill

## Core Philosophy

Structure generation follows a **funnel process**: start broad (many compositions, chemistries, configurations), then narrow using physical constraints (charge neutrality, Ewald energy, thermodynamic stability). The workflow is modular—enter at any phase and skip steps that don't apply.

**Complete pipeline:**
```
Phase 1: Compositions → Seed Structures (if no template exists)  
Phase 2: Seed Structures → Chemical Variants (substitution, doping, ion exchange)
Phase 3: Ordered/Disordered → Resolved Structures (enumeration, SQS, majority-species)
Phase 4: Structures → Defect Supercells (vacancies, substitutions, interstitials)
Phase 5: Structures → Perturbed Structures (rattling, strain for DFT/ML initialization)
Phase 6: All → Per-candidate directory storage + ASE database index
```

**Entry points match your starting information:**
- Elements only (Li-Mn-P-O)? → Start Phase 1
- Composition known (LiMnPO₄) but no structure? → **Search both `mp_search_materials` and `cod_search_structures`** for a seed before falling back to prototype building.
- Structure from MP/COD/CIF/ASE exists? → Start Phase 2 (or skip to Phase 3-5)

**Working principles:**
1. **MCP tools generate complete structures** — Every tool returns CIF/POSCAR with atom positions, unit cell, spacegroup. Custom formula generators without structures cannot be validated or screened.
2. **Organize candidates in a `candidates/` directory** — Each candidate gets its own subdirectory named clearly (e.g., `candidates/005_RbGaS2/`). This directory stores provenance metadata, unrelaxed structures, relaxed structures, and any downstream data for that candidate.
3. **ASE database is central index** — Use ASE database for cross-candidate queries and property enrichment, but keep the canonical outputs in the per-candidate directory structure.
4. **Metadata enables screening** — Tag structures with `requires_ordering`, `doping_concentration`, `host_formula` so candidate-screener knows how to handle disorder.
5. **Scale requires planning** — For N > 20 candidates, create a planning file first to avoid tool timeout/memory issues (see [references/phase-guides.md](references/phase-guides.md)).

> **Why use MCP tools instead of scripts?** Because materials discovery needs real crystal structures (atom positions + lattice parameters) to:
> - Validate geometry (coordination, bond lengths) before expensive calculations
> - Run ML predictions (MatGL) or DFT (VASP) on actual atomic configurations
> - Check thermodynamic stability (energy above hull from Materials Project API)  
> - Export to synthesis planning (experimental precursor prediction needs spatial arrangement)
>
> A formula string like "LiₓCoO₂" without atomic positions is scientifically invalid for screening

---

## Workflow Phases (High-Level)

**For complete detailed instructions, algorithms, parameter tables, and decision logic, see [references/phase-guides.md](references/phase-guides.md).**

### Phase 1: Seed Structure Building

**When:** Need to build structure from scratch (no MP/COD/CIF template available)  
**Skip if:** Structure already exists

**Before falling back to prototype building, always search both databases:**
1. `mp_search_materials(formula=...)` — rich DFT properties available if found
2. `cod_search_structures(formula=...)` — fills the gap for experimental/niche compounds not in MP

Only reach for `pymatgen_prototype_builder` when neither database has the compound.

**Common prototypes:** Rock-salt (225), Perovskite (221), Spinel (227), Layered oxide (166), Olivine (62)

**Output:** Crystal structure from spacegroup + species + lattice parameters

**Next:** Phase 2 (chemical exploration)

---

### Phase 2: Chemical Space Exploration

**When:** Want to explore composition/doping variants while preserving structure type  
**Skip if:** Want to keep exact composition

**Two branches:**

**Branch A — Charge-neutral ionic substitution:**
- Tool: `pymatgen_ion_exchange_generator`
- Use for: Battery materials, ionic conductors where charge balance is critical
- Automatically adjusts stoichiometry to maintain neutrality
- Example: Li → Na substitution in LiCoO₂ → NaCoO₂

**Branch B — Exploratory screening:**
- Tool: `pymatgen_substitution_generator`  
- Use for: Isostructural analogues, ML training sets, broad exploration
- Generates ordered configurations with integer occupancy
- Example: Screen Ni/Mn/Co mixing in layered oxides

**Decision:** Ionic material + charge-balancing is critical? → Branch A. Exploratory screening? → Branch B.

**Output:** 1-100 chemical variants (all real structures, not formulas)

**Next:** Phase 3 (if disorder present), Phase 4 (defects), Phase 5 (perturbation), or storage

---

### Phase 3: Disorder Resolution

**When:** Structures have fractional site occupancies (e.g., Sr₀.₉₇Sm₀.₀₃NbO₄)  
**Skip if:** All structures fully ordered

**Two directions:**

**Creating disorder** (ordered → fractional):
- Tool: `pymatgen_disorder_generator`
- Use for: Statistical doping, dilute substitution, solid solutions
- Creates fractional occupancy on sites (not ordered supercell variants)

**Resolving disorder** (fractional → ordered):
Choice depends on **doping concentration**:

| Concentration | Tool | Physical Basis | Use Case |
|--------------|------|----------------|----------|
| < 10% | `pymatgen_majority_orderer` | Host dominates, minority negligible | Fast screening (unit cell) |
| 10-20% | Majority (screen) → SQS (validate) | Approximation acceptable for filtering | Two-stage: cheap screen + accurate validation |
| > 20% | `pymatgen_sqs_orderer` | Dopant-dopant interactions matter | Solid solutions, high-entropy (supercell) | 

**Why concentration matters:** At low doping, dopants are isolated (host lattice determines properties). At high concentration, dopant-dopant interactions and local ordering become important (need supercell to capture correlations). Screening workflows balance speed (majority orderer on unit cell, seconds) vs accuracy (SQS on 200-atom supercell, minutes).

**Important:** When generating disordered structures for screening, attach metadata:
```python
metadata = {
    "requires_ordering": "majority" | "enumeration" |"sqs",  
    "doping_concentration": 0.03,  # For physical validity assessment
    "host_formula": "SrNb2O6",
    "dopant_species": ["Sm"]
}
```

This tells candidate-screener how to preprocess before validation.

**Output:** Ordered structures ready for DFT/ML screening

**Next:** Phase 4 (defects), Phase 5 (perturbation), or storage

---

### Phase 4: Defect Generation

**When:** Need point defects (vacancies, substitutions, interstitials) for defect chemistry studies  
**Skip if:** Perfect crystals only

**Tool:** `pymatgen_defect_generator`

**Three defect types:**
- Vacancies: Remove atoms (e.g., oxygen vacancies in oxide catalysts)
- Substitutional: Replace atom with different species (e.g., Al on Si site in semiconductors)
- Interstitial: Insert atom in unoccupied position (e.g., Li in diffusion path studies)

**Output:** Supercells (2×2×2 typical) with defects

**Next:** Phase 5 (perturbation for relaxation initialization) or storage

---

### Phase 5: Perturbation (Rattling + Strain)

**When:** Initializing structures for DFT/ML relaxation, generating ML training diversity, or exploring metastable configurations  
**Skip if:** Using unperturbed geometries

**Tool:** `pymatgen_perturbation_generator`

**Two perturbation types:**
- Atomic displacement: Rattle atoms by 0.01-0.1 Å (breaks symmetry for DFT initialization)
- Lattice strain: Apply hydrostatic/directional strain ±1-5% (initializes cell relaxation, generates diverse configurations for ML training)

**Parameter guidance by use case:**
- DFT initialization: 0.01 Å displacement + 1% strain (small, helps convergence)
- ML training diversity: 0.05-0.1 Å + 2-5% strain (larger, samples configuration space)
- Metastable search: 0.1 Å + directional strain (trigger phase transitions)

**Output:** Perturbed structures ready for relaxation

**Next:** Storage (ASE database)

---

### Phase 6: Output Organization & Storage

**Directory structure — organize each candidate in its own subdirectory:**
```
candidates/
├── 001_LiCoO2/
│   ├── 001_LiCoO2.cif              # Generated structure
│   ├── metadata.json               # Generation metadata (tool, parent, params)
│   ├── 001_LiCoO2_relaxed.cif      # Post-relaxation (added later by screener)
├── 002_NaCoO2/
│   ├── 002_NaCoO2.cif
│   └── metadata.json
└── ...
```

Naming convention: `NNN_Formula` where `NNN` is a zero-padded index and `Formula` is a clear, human-readable reduced formula. This makes candidates easy to reference across tools and scripts.

**ASE database as cross-candidate index:** Store generated structures in ASE for downstream retrieval, property enrichment, and batch filtering. Convert CIF/POSCAR outputs with the dedicated ASE conversion step, then store with metadata:

**Essential metadata fields:**
```python
metadata = {
    # Generation info
    "generation_phase": "phase_2_substitution",
    "parent_structure_id": "mp-1234",
    "generation_tool": "pymatgen_substitution_generator",
    
    # Chemical info
    "formula": "LiNi0.8Mn0.2O2",
    "host_formula": "LiNiO2",  
    "elements": ["Li", "Ni", "Mn", "O"],
    
    # Disorder info (if applicable)
    "is_disordered": False,
    "requires_ordering": "none",
    "doping_concentration": 0.20,
    
    # Purpose
    "screening_target": "battery_cathode",
    "batch_id": "cathode_screen_001"
}
```

**Best practice:** Store after each phase so intermediate structures are recoverable.

**Example:**
```python
ase_store_result(
    db_path="candidates.db",
    atoms_dict=structure_cif,  # From any MCP output
    key_value_pairs=metadata
)
```

---

## Tool Reference (Condensed)

**The condensed table below should be the default reference. If a parameter detail matters, inspect the tool schema directly rather than relying on a duplicated local catalog.**

| Phase | Tool | Purpose | Key Decision |
|-------|------|---------|--------------|
| 0 | `pymatgen_substitution_predictor` | ICSD-guided substitutions | Uses lambda-scaling |
| 0 | `mp_search_materials` | Template structures (DFT) | Elements, stability filter |
| 0 | `cod_search_structures` | Template structures (experimental) | Niche/niche compounds not in MP |
| 1 | `pymatgen_prototype_builder` | Build from spacegroup | Need lattice parameter estimate |
| 2A | `pymatgen_ion_exchange_generator` | Charge-neutral substitution | Auto stoichiometry adjustment |
| 2B | `pymatgen_substitution_generator` | Ordered enumeration | Integer occupancy |
| 3 | `pymatgen_disorder_generator` | Create fractional occupancy | Statistical doping |
| 3 | `pymatgen_majority_orderer` | Dilute doping approximation | < 10% concentration |
| 3 | `pymatgen_enumeration_orderer` | Exhaustive configurations | 10-30% concentration |
| 3 | `pymatgen_sqs_orderer` | Stat quasirandom structures | > 20% solid solutions |
| 4 | `pymatgen_defect_generator` | Point defects | Vacancy, substitution, interstitial |
| 5 | `pymatgen_perturbation_generator` | Rattle + strain | DFT init vs ML diversity |
| 6 | `ase_store_result` | Database storage | Convert CIF/POSCAR outputs before storage |

**Key tool distinction:**

**`disorder_generator` vs `substitution_generator`:**

| Aspect | disorder_generator | substitution_generator |
|--------|-------------------|----------------------|
| **Occupancy** | Fractional (80% Ni + 20% Mn on same site) | Integer (one site 100% Mn, others 100% Ni) |
| **Output count** | 1 statistically disordered structure | Multiple ordered configurations |
| **Use for** | Dilute doping (<10%), SQS generation | Supercell enumeration, DFT screening |
| **Example** | Li[Ni₀.₈Mn₀.₂]O₂ → every TM site has fractional occupancy | Li[Ni₀.₈Mn₀.₂]O₂ → 10-atom supercell with 8 Ni sites, 2 Mn sites |

**Rule:** For partial substitution, ask: Do you want statistical (every site has fractional occupancy) or configurational (specific ordered supercells)? Statistical → `disorder_generator`. Configurational → `substitution_generator`.

---

## Decision Trees

**For extended decision logic, scaling notes, and phase-specific tradeoffs, see [references/phase-guides.md](references/phase-guides.md).**

### Strategy Selection (by starting information)

```
What do you have?
├─ Composition (LiCoO₂) → Check MP for structure
│   ├─ Found on MP → Phase 2 (chemical exploration)
│   └─ Not found → Phase 1 (prototype building)
├─ Structure (CIF/POSCAR) → Phase 2/3/4/5 depending on goal
│   ├─ Want chemical variants? → Phase 2
│   ├─ Has disorder? → Phase 3
│   ├─ Want defects? → Phase 4
│   └─ Want perturbations? → Phase 5
└─ Structure set (ASE database) → Batch operations on all
```

### Disorder Strategy Selection

```
Is structure disordered?
├─ NO → Continue to screening or next phase
├─ YES → What's the doping concentration?
    ├─ < 10% (dilute) → disorder_generator → majority_orderer
    │   └─ Fastest path (unit cell), valid for screening
    ├─ 10-20% (intermediate) → disorder_generator → majority (screen) + SQS (validate)
    │   └─ Two-stage: cheap screening, accurate validation of top 10
    └─ > 20% (solid solution) → sqs_orderer directly
        └─ Supercell required, but most accurate for high concentration
```

---

## Common Pitfalls

**See [references/phase-guides.md](references/phase-guides.md) for deeper workflow tradeoffs and [references/examples.md](references/examples.md) for full working patterns.**

1. **Generating formula strings instead of structures**  
   ❌ `formulas = [f"Li{x}CoO2" for x in [0.5, 1.0, 1.5]]` — No atomic positions, can't screen  
   ✅ Use `pymatgen_ion_exchange_generator` → outputs real CIF files

2. **Forgetting to attach metadata for screening**  
   If generating disordered structures, screening will crash without `requires_ordering` metadata. Attach classification at generation time.

3. **Using wrong tool for concentration**  
   ❌ `sqs_orderer` for 3% doping → Massive supercell (200 atoms), 50× slower, unnecessary  
   ✅ Use `disorder_generator` → `majority_orderer` for dilute doping

4. **Not storing intermediate phases**  
   If Phase 1 → Phase 2 → Phase 3 workflow crashes at Phase 3, you lose Phase 1/2 results. Store after each phase in ASE database.

5. **Skipping charge neutrality checks**  
   For ionic materials, charge-imbalanced structures are unphysical. Use `ion_exchange_generator` or manually verify with `composition_analyzer`.

---

## Complete Workflow Examples

**For 9 full working patterns (from simple isostructural screens to complex multi-phase pipelines), see [references/examples.md](references/examples.md).**

**Pattern catalog:**
- Pattern 1: Isostructural analogue screen (simple, Phase 2 only)
- Pattern 2: Battery cathode analogue (Phase 2 ion exchange)
- Pattern 3: High-entropy oxide SQS (Phase 1 → Phase 3)
- Pattern 4: Ground-state ordering search (Phase 3 enumeration)
- Pattern 5: Lanthanide-doped phosphor screen (Phase 1 → Phase 2 → Phase 3, 93 structures)
- Pattern 6: Oxygen vacancy defects (Phase 4)
- Pattern 7: Perovskite B-site doping (Phase 2B)
- Pattern 8: ML training set generation (Phase 2 + Phase 5)
- Pattern 9: Full pipeline (Phase 1 → Phase 5, all capabilities)

**Example: Simple isostructural rocksalt screen (10 structures in 30 seconds):**

```python
# Goal: Screen Li-O, Na-O, K-O, Mg-O rocksalt structures

# Get rocksalt template (LiF)
mp_result = mp_search_materials(formula="LiF", limit=1)
lif_structure_cif = mp_result['structures'][0]

# Phase 2: Substitute Li with Na, K, Mg
result = pymatgen_substitution_generator(
    input_structures=lif_structure_cif,
    substitutions={'Li': ['Na', 'K', 'Mg']},
    n_structures=1,  # One ordered structure per substitution
    enforce_charge_neutrality=True
)

# Storage: Save all 3 structures
for i, structure_cif in enumerate(result['structures']):
    ase_store_result(
        db_path="rocksalt_analogues.db",
        atoms_dict=structure_cif,
        key_value_pairs={
            "formula": result['metadata'][i]['formula'],
            "generation_phase": "phase_2_substitution",
            "screening_target": "rocksalt_ionic_conductors"
        }
    )

# Next: Pass database to candidate-screener for property predictions
```

For more complex examples including disorder handling, defects, and multi-phase workflows, see [references/examples.md](references/examples.md).

---

## Connection to Screening Workflow

**After generation, pass structures to candidate-screener skill for:**
1. Validation (`structure_validator`) — Check coordination, bond lengths, stereospatial issues
2. Preprocessing (if disordered) — Apply orderers based on `requires_ordering` metadata  
3. Property enrichment (MatGL, matcalc) — Formation energy, band gap, elasticity, phonons
4. Ranking (multi-objective) — Score by stability, property targets, synthesis likelihood

**Hand-off format:** ASE database path or list of CIF strings

**Important:** Attach metadata during generation (especially `requires_ordering` for disordered structures) so screener knows how to preprocess.

**See also:** candidate-screener skill for complete validation and ranking workflows.

---

## Additional Resources

**All detailed documentation in references/ directory:**

- **Batch scripts** — Use `matclaw_sdk` (install with `pip install -e /path/to/MatClaw/sdk/`). Import tools directly: `from matclaw_sdk import tool_name`. See [examples/batch_generation_example.py](examples/batch_generation_example.py) for a complete example.
- **[references/phase-guides.md](references/phase-guides.md)** — Complete Phase 1-5 instructions with algorithms, parameter tables, decision logic, physical basis explanations
- **[references/examples.md](references/examples.md)** — 9 complete working patterns from simple to complex, including full code and expected outputs
