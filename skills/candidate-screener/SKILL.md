---
name: candidate-screener
description: >
  Transform candidate structures into ranked, property-enriched, synthesis-ready materials through intelligent validation, hierarchical property retrieval, and multi-objective optimization. Validates structure integrity, retrieves properties via intelligent hierarchy (Materials Project DFT → ASE cache → ML prediction with MatGL/matcalc), handles disordered structures through preprocessing layer (majority/enumeration/SQS ordering), applies application-specific screening criteria, and ranks by confidence-weighted multi-objective methods. Core strength: Complete transparency - tracks rejection reasons, flags ML uncertainties for DFT verification, implements confidence scoring, and provides complete provenance. Supports all screening workflows: battery cathodes (voltage, capacity, stability), catalysts (surface energies, adsorption), thermoelectrics (band gap, transport), phosphors (optical properties, rare-earth incorporation), mechanical materials (elasticity, phonons), and custom multi-criteria screening. When NOT to trigger: simple property lookup without ranking/filtering (use MP/ASE tools directly), VASP/DFT result analysis (use vasp-ase skill), active learning with Bayesian optimization (use active-learning skill).
  
  Trigger keywords: screen candidates, validate structures, property retrieval, hierarchical data, rank materials, filter candidates, multi-objective optimization, battery screening, catalyst discovery, thermoelectric materials, phosphor screening, mechanical properties, phonon stability, surface energies, formation energy prediction, band gap prediction, disordered structures, dilute doping, solid solutions, materials discovery pipeline, high-throughput screening, candidate ranking, confidence scoring, MP + ML hybrid, MatGL predictions, matcalc calculations, structure preprocessing, ordering strategies, synthesis-ready candidates.
---

# Candidate Screener Skill

Validates and enriches candidate structures with properties using hierarchical data retrieval (Materials Project → ASE cache → ML prediction), applies application-specific screening criteria, and ranks by multi-objective optimization.

**Input:** List of candidate structures from candidate-generator skill  
**Output:** Ranked list of property-enriched candidates ready for synthesis/DFT validation

---

## Core Philosophy

**This skill implements a property enrichment pipeline, NOT a blind filter:**

1. **Hierarchical property retrieval** (MP → ASE cache → ML calculation)
   - Try high-quality DFT data first (Materials Project — includes band gap, formation energy)
   - Then cached results (ASE database)
   - Only run ML calculations if necessary (MatGL + matcalc)

2. **Complete transparency** - log rejection reasons for diagnostics and refinement
   - Invalid structures: rejected with cause
   - ML prediction failures: flagged for DFT verification
   - Failed criteria: recorded with specific thresholds

3. **Confidence-aware ranking**
   - MP (DFT) properties: confidence = 1.0
   - ASE cached: confidence = 0.8-1.0 (depends on source)
   - ML predictions: confidence = 0.65-0.75
   - High-scoring ML predictions flagged for DFT verification

4. **Two ML ecosystems:**
   - **MatGL** (direct predictions): Fast formation energy + band gap (~0.5-1s each)
   - **matcalc** (structure-based calculations): Mechanical, vibrational, surface, thermal properties (~20-60s each)
   - Relax structures first (ML models trained on equilibrium geometries, 5-10s)

---

## Quick Tool Reference

### Phase 1: Validation & Analysis
| Tool | Purpose | Speed |
|------|---------|-------|
| `structure_validator` | Check structure integrity | 0.1s |
| `composition_analyzer` | Analyze composition, oxidation states | 0.05s |
| `mp_search_materials` + `mp_get_material_properties` | MP-backed stability lookup primitives; usually invoked through the `stability-analyzer` skill | 0.5-2s |
| `structure_analyzer` | Compute symmetry, coordination | 0.1s |
| `structure_fingerprinter` | Detect duplicates | 0.5s |

### Phase 2A: Property Retrieval (Hierarchical)
| Source | Tools | Priority | Confidence |
|--------|-------|----------|------------|
| Materials Project | `mp_search_materials`, `mp_get_material_properties` | 1st (best) | 1.0 (DFT) |
| ASE cache | `ase_query`, `ase_connect_or_create_db` | 2nd | 0.8-1.0 |
| ML calculation | MatGL + matcalc | 3rd | 0.65-0.75 |

### Phase 2B: ML Calculations
| Ecosystem | Tools | Properties | Speed |
|-----------|-------|------------|-------|
| **MatGL** | `matgl_predict_eform`, `matgl_predict_bandgap` | Formation energy, band gap | 0.5-1s |
| **matcalc** | `matcalc_calc_elasticity`, `matcalc_calc_phonon`, `matcalc_calc_surface`, etc. | Mechanical, vibrational, surface, thermal | 20-60s |
| **Relaxation** | `matgl_relax_structure` | Before all predictions (removes artifacts, handles disorder) | 5-10s |

**MatGL tools (fast screening):**
- `matgl_predict_eform`: Formation energy (M3GNet/MEGNet 2018 models)
- `matgl_predict_bandgap`: Electronic band gap (MEGNet 2019 model)

**matcalc tools (detailed calculations, 2025 ML potentials):**
- `matcalc_calc_elasticity`: Elastic tensor, bulk/shear/Young's modulus
- `matcalc_calc_phonon`: Phonon dispersion, dynamic stability
- `matcalc_calc_surface`: Surface energies (catalyst screening)
- `matcalc_calc_eos`: Equation of state, bulk modulus
- `matcalc_calc_adsorption`: Adsorption energies
- `matcalc_calc_md`: Molecular dynamics
- `matcalc_calc_neb`: Reaction barriers, diffusion paths
- `matcalc_calc_phonon3`: Thermal conductivity
- `matcalc_calc_qha`: Thermal expansion
- `matcalc_calc_energetics`: Formation + cohesive energy
- `matcalc_calc_interface`: Grain boundary / heterostructure energies

### Phase 3: Ranking & Selection
| Tool | Purpose | Speed |
|------|---------|-------|
| `multi_objective_ranker` | Rank by multiple criteria (Pareto/weighted sum) | 10s |

**Storage:**
- `ase_store_result`: Cache results in ASE database for future runs

**Use the condensed table below for normal operation. If parameter details matter, inspect the tool schema directly instead of maintaining a parallel local catalog.**

---
## Preprocessing: Disorder Resolution (Step 0)

**Why preprocessing matters:** MatGL/matcalc use ASE, which cannot represent partial site occupancies. Disordered structures (fractional occupancy from doping, solid solutions) require conversion to fully ordered structures before screening because ASE's Atoms object only supports integer site occupancies.

**When disordered structures appear:**
- `candidate-generator` outputs (disorder_generator, ion_exchange_generator)  
- Materials Project rare-earth / mixed-valence compounds
- User-provided CIF files with crystallographic disorder

**Three ordering strategies:**

| Strategy | Tool | Use Case | Validity Range | Speed |
|----------|------|----------|----------------|-------|
| **Majority** | `pymatgen_majority_orderer` | Dilute doping, fast screening | < 10% dopant valid, 10-20% questionable | Fastest (1 structure) |
| **Enumeration** | `pymatgen_enumeration_orderer` | Site-specific studies, exhaustive exploration | All configurations valid | 10-50× structures |
| **SQS** | `pymatgen_sqs_orderer` | Solid solutions, high-entropy, > 20% mixing | Most accurate for concentrated disorder | Slow (large supercells) |

**Decision logic:**
```
IF structure.is_ordered → Skip preprocessing
ELIF metadata.requires_ordering exists → Use specified strategy
ELIF doping_concentration < 10% → majority_orderer (safe default)
ELIF doping_concentration > 20% → SQS ordering (flag if generator didn't provide SQS metadata)
ELSE → majority_orderer with DFT validation flag
```

**Validation flags after preprocessing:**
```python
candidate['preprocessing_metadata'] = {
    "was_disordered": True,
    "strategy": "majority",
    "approximation_valid": (doping < 0.10),
    "requires_dft_validation": (doping >= 0.10)
}
```

**For complete decision trees, implementation examples, and physical basis explanations, see [references/preprocessing-guide.md](references/preprocessing-guide.md)**


**Use in ranking (Step 4):**
- Structures with `approximation_valid=False` get lower confidence scores
- Flag high-scoring ML predictions for DFT verification (validates approximations before synthesis investment)

---
## Universal Essential Properties

**Essential properties for all materials screenings (baseline characterization):**

### 1. Validation (Phase 1)
- ✅ **Structure valid**: No overlapping atoms, valid geometry
- ✅ **Composition valid**: Charge-neutral, reasonable oxidation states
- ⚠️ **Thermodynamic stability**: Use cheap known-data checks only at this stage
   Use the `stability-analyzer` skill here only in its MP-backed lookup route for known Materials Project compositions or fast sanity checks, not for custom hull construction.

### 2. Basic Properties (Phase 2)
- ✅ **Formation energy**: Thermodynamic stability indicator (negative = stable relative to elements)
  - **Source priority:** MP → ASE → `matgl_predict_eform` (fast)
  - **NOT** `matcalc_calc_energetics` (20× slower, use only for cohesive energy)

- ⚠️ **Energy above hull** (optional, computationally intensive)
   If the screening criteria include `energy_above_hull`, evaluate it in Phase 2 and refer to the `stability-analyzer` skill. That skill decides whether the request should stay a cheap MP-backed lookup or escalate to a custom self-consistent hull workflow. Reserve Phase 1 for cheaper validation and MP-backed checks.

- ⚠️ **Band gap** (optional): Electronic properties
  - **Source priority:** MP → ASE → `matgl_predict_bandgap` (only tool)

### 3. Application-Specific Properties (Phase 2, conditional)
Choose based on application:

**Battery cathodes:** Band gap, mechanical stability  
**Catalysts:** Surface energies, adsorption energies  
**Thermoelectrics:** Band gap, thermal conductivity  
**Structural materials:** Mechanical properties (elasticity, hardness)  
**Phosphors:** Band gap, optical properties

---

## Workflow Execution

**5-phase algorithm (expanded workflow, failure handling, and batch execution in [references/execution-guide.md](references/execution-guide.md)):**

1. **Preprocessing (Step 0)**: Resolve disordered structures (see decision table above)
2. **Validation (Phase 1)**: Check structure validity, composition, stability, deduplicate
3. **Property Retrieval (Phase 2)**: Hierarchical lookup (MP → ASE → ML), relax before predictions, cache all results
4. **Filtering (Phase 3)**: Apply screening criteria, reject with specific reasons
5. **Ranking (Phase 4)**: Multi-objective optimization, confidence-weighted scoring, flag ML predictions for DFT verification

**Essential principles:**
- Follow hierarchy order (MP → ASE → ML preserves data quality, DFT > ML approximations)
- Relax structures before ML predictions (ML models trained on equilibrium geometries)
- Cache all results in ASE database (prevents redundant calculations in future runs)
- Save relaxed structures (needed for DFT validation and synthesis planning)
- Track all exclusions with reasons (enables criteria refinement and debugging)

---

## Tool Selection Guides

**Hierarchical retrieval priority:** MP (DFT-quality) → ASE cache (instant) → ML (fast approximation)  
Follow hierarchy order - DFT data quality is worth the wait because it eliminates downstream validation needs.

**MatGL vs matcalc decision:**

| Use Case | Tool | Speed | When |
|----------|------|-------|------|
| Formation energy screening | MatGL | ~0.5s | Initial rapid filtering (100+ candidates) |
| Band gap screening | MatGL | ~0.5s | Initial rapid filtering |
| Mechanical properties | matcalc | ~20-60s | Top 10-20 after MatGL filtering |
| Thermal/vibrational | matcalc | ~20-60s | Final detailed analysis |
| Surface properties | matcalc | ~20-60s | Catalyst applications |

**Strategy:** MatGL screens 100 → 40 (minutes), then matcalc analyzes top 20 (hours).  
See [references/ml-calculations-guide.md](references/ml-calculations-guide.md) for complete usage guide.

---

## Large-Scale Screening

**For >20 candidates:**
1. Create checkpoint-based tracking JSON (enables resume after interruptions)
2. Present screening plan to user (candidates, criteria, runtime estimate)
3. Execute with save-after-every-candidate pattern
4. Support iterative refinement (adjust criteria, reuse cached properties)

**Batch scripts:** Use `matclaw_sdk` (install with `pip install -e /path/to/MatClaw/sdk/`). Import tools directly: `from matclaw_sdk import tool_name`. See [examples/batch_screening_example.py](examples/batch_screening_example.py) for a complete example.  
Complete implementation is in [references/execution-guide.md](references/execution-guide.md).

---

## Critical Decisions

**Four key decision algorithms:**

| Decision | Logic | Details |
|----------|-------|---------|
| Structure relaxation | Skip if DFT/MP/experimental source, else relax | references/execution-guide.md |
| ML failure handling | Try M3GNet → MEGNet → MP similarity → flag for DFT | Track all failures for diagnosis |
| Multiple MP matches | Keep all if exploring metastable, else most stable | Polymorph awareness |
| Confidence weighting | MP: 1.0, ASE: 0.8-1.0, ML: 0.65-0.75 | Flag high-score ML for DFT |

---

## Output Structure

**Screening report includes:**
- Summary stats (input, validated, with_properties, passed, ranked, runtime)
- Data source breakdown (MP/ASE/ML counts)
- Top candidates (with original + relaxed structures, properties, scores, DFT recommendations)
- Rejected candidates (with specific failure reasons)
- Property distributions (min/max/mean for each property)

**Structure preservation:** Original CIF (provenance) + relaxed CIF (DFT input/synthesis) both saved.

**Performance:** 100 candidates = 2 min (80% MP) to 20 min (100% ML).

---

## Integration

**Input from candidate-generator:** Direct JSON file feed with structures.  
**Output to synthesis-planner:** Top-ranked candidates with relaxed structures and properties.

---

## Reference Documentation

**Core references in `references/` directory:**

1. **[preprocessing-guide.md](references/preprocessing-guide.md)** - Disorder handling, ordering strategies, physical basis
2. **[execution-guide.md](references/execution-guide.md)** - Workflow order, decision branches, failure handling, and large-batch execution
3. **[ml-calculations-guide.md](references/ml-calculations-guide.md)** - MatGL vs matcalc usage guide, model selection, parameter tuning
4. **Use the dedicated `stability-analyzer` skill** - When screening criteria include stability or `energy_above_hull`, especially if the workflow must choose between MP-backed lookup and custom self-consistent hull analysis

**Read these references when:**
- Resolving disordered inputs before screening
- Implementing or debugging the end-to-end screening workflow
- Choosing between MatGL and matcalc for a property
- Adding `energy_above_hull` as a screening criterion for candidates without a direct known MP hull value

---

**Last updated:** 2025-01-28  
**Skill version:** 2.0 (refactored with progressive disclosure)
