---
name: candidate-screener
description: >
  Guide for screening and ranking materials candidates — validating structures, retrieving properties, and ranking by multi-objective criteria. Helps decide how to handle disordered structures (majority/enumeration/SQS ordering), suggests where to get properties (Materials Project, ASE cache, or ML predictions with MatGL/matcalc), and ranks candidates with confidence-aware multi-objective methods. Use this whenever candidates need property enrichment and ranking: validating candidate structures, predicting or retrieving formation energy / band gap / mechanical / vibrational / surface properties, filtering by application-specific criteria, or ranking by multiple objectives.
  
  Trigger keywords: screen candidates, validate structures, property retrieval, hierarchical data, rank materials, filter candidates, multi-objective optimization, battery screening, catalyst discovery, thermoelectric screening, phosphor screening, mechanical properties, phonon stability, surface energies, formation energy prediction, band gap prediction, disordered structures, dilute doping, solid solutions, high-throughput screening, candidate ranking, confidence scoring, MatGL predictions, matcalc calculations, structure preprocessing, ordering strategies, synthesis-ready candidates.
---

# Candidate Screener Skill

Helps validate, enrich with properties, and rank candidate materials for downstream synthesis or DFT validation.

**Input:** Candidate structures (CIFs, often from `candidate-generator`)  
**Output:** Ranked, property-enriched candidates for synthesis or further analysis

---

## How to think about screening

Screening is **enrichment plus ranking**, not a blind filter. The goal is to attach enough information to each candidate so you and the user can make informed decisions about which ones to pursue. At a high level, screening involves:

- **Validating** that structures are physically reasonable
- **Retrieving or predicting properties** relevant to the application
- **Ordering disordered structures** when needed (ASE tools can't represent fractional occupancy)
- **Filtering and ranking** by the criteria that matter

How you combine these steps depends on the research question, the number of candidates, and what properties are needed. There's no single mandatory sequence — design the approach that fits the task.

---

## Quick Tool Reference

### Structure Analysis & Validation
| Tool | Purpose | Speed |
|------|---------|-------|
| `structure_validator` | Check structure integrity | 0.1s |
| `composition_analyzer` | Analyze composition, oxidation states | 0.05s |
| `structure_analyzer` | Compute symmetry, coordination | 0.1s |
| `structure_fingerprinter` | Detect duplicate structures | 0.5s |
| `mp_search_materials` | MP stability lookup (often via `stability-analyzer`) | 0.5-2s |

### Property Retrieval & Prediction
| Source | Tools | Confidence | Speed |
|--------|-------|------------|-------|
| **Materials Project** | `mp_search_materials`, `mp_get_material_properties` | Highest (DFT) | 0.5-2s |
| **ASE cache** | `ase_query`, `ase_connect_or_create_db` | High (cached) | Instant |
| **MatGL (ML)** | `matgl_predict_eform`, `matgl_predict_bandgap` | Moderate | 0.5-1s |
| **Relaxation** | `matgl_relax_structure` | Needed before ML predictions | 5-10s |
| **matcalc (ML)** | `matcalc_calc_elasticity`, `matcalc_calc_phonon`, `matcalc_calc_surface`, `matcalc_calc_eos`, `matcalc_calc_adsorption`, `matcalc_calc_md`, `matcalc_calc_neb`, `matcalc_calc_phonon3`, `matcalc_calc_qha`, `matcalc_calc_energetics`, `matcalc_calc_interface` | Moderate | 20-60s |

### Ranking & Persistence
| Tool | Purpose | Speed |
|------|---------|-------|
| `multi_objective_ranker` | Rank by multiple criteria (Pareto/weighted sum/constraint) | 10s |

---

## Guiding principles

These aren't requirements — they're reasoning to help you design an effective screening.

### Properties: start with the best available source, but use judgment

Higher-confidence data (DFT > cached > ML) is generally better, but the right approach depends on the task:

- **Materials Project** gives DFT-quality data instantly for known compositions — great for band gaps, formation energies, and energy above hull. Consider checking it first when applicable.
- **ASE cache** stores results from previous runs — worth checking if properties were already computed.
- **MatGL predictions** (`matgl_predict_eform`, `matgl_predict_bandgap`) are fast (~0.5s) and suitable for screening many candidates where DFT data isn't available.
- **matcalc calculations** (elasticity, phonons, surfaces, etc.) are slower (~20-60s) so best reserved for top candidates after initial filtering.
- **Relax structures** before ML predictions if they don't come from a trusted optimized source (DFT/MP/experiment) — ML models perform best on near-equilibrium geometries.

You can skip, reorder, or add steps as the situation demands. If the user already has MP data, there's no need to re-fetch it. If they want ML-only screening, that's fine too. Clarify with the user when you're unsure which source to prioritize.

### Keep a transparent record

For any candidate that gets filtered out or fails, note why. This helps the user refine their criteria and makes the screening auditable. Track:
- Which structures were rejected and why (invalid geometry, failed prediction, etc.)
- What property source was used for each value (MP, ASE cache, MatGL, matcalc)
- Which candidates had approximations applied (e.g., majority ordering of a disordered structure)

### Confidence-aware ranking

When ranking, account for data provenance:
- MP (DFT) values can be treated as ground truth
- ASE cached values are reliable but may use different methods
- ML-predicted values are useful for trends but less reliable for absolute numbers
- High-scoring ML-only candidates may warrant DFT verification before synthesis investment

`multi_objective_ranker` supports Pareto optimization, weighted-sum scoring, and constraint-based filtering — choose the strategy that matches the user's goals.

---

## Handling disordered structures

ASE-based tools (MatGL, matcalc) require fully ordered structures — each site needs exactly one atom type. If your candidates have fractional occupancies (doping, solid solutions, mixed-valence compounds), they need to be converted before property prediction.

### Three ordering strategies

| Strategy | Tool | Best for | What it produces |
|----------|------|----------|-----------------|
| **Majority** | `pymatgen_majority_orderer` | Low doping where the host lattice dominates | 1 structure (keep dominant species per site) |
| **Enumeration** | `pymatgen_enumeration_orderer` | Exhaustive exploration of site arrangements | 10-50 structures (all symmetry-distinct orderings) |
| **SQS** | `pymatgen_sqs_orderer` | Concentrated solid solutions, high-entropy alloys | 1-5 large supercells mimicking random disorder |

### How to choose

- **Low doping (<5%):** `majority_orderer` is a good fit — the host lattice dominates, so keeping the majority species per site preserves the essential physics.
- **Higher doping (≥5%):** You have a choice. **Enumeration** is more thorough — it generates all symmetry-distinct configurations and ranks them by Ewald energy — but produces many structures. **SQS** gives one representative supercell that approximates random disorder, which is more accurate for concentrated solid solutions but creates a larger cell.
  
  The right pick depends on the research question: enumeration if you want to explore all possible site arrangements, SQS if you want a single realistic model of a random solid solution. Consult the user if you're unsure.

### When user-provided metadata exists

If the `candidate-generator` or input metadata specifies a strategy (`majority`, `enumeration`, `sqs`), honor that choice — it was made deliberately.

### Flagging approximations

After ordering, attach metadata so subsequent steps can account for it:
```python
candidate["preprocessing_metadata"] = {
    "was_disordered": True,
    "strategy": "majority",
    "approximation_valid": (doping < 0.05),  # 5% threshold
}
```
Approximated structures may get slightly lower confidence in ranking, and high-scoring ones can be flagged for DFT verification.

**For full decision trees, implementation details, and physical basis, see [references/preprocessing-guide.md](references/preprocessing-guide.md)**

---

## Large-scale screening (>20 candidates)

For larger batches, consider:

1. **Checkpointing** — Save progress after each candidate so screening can resume if interrupted. A tracking JSON with candidate status, property sources, and errors works well.
2. **Phased execution** — Start with fast validation and MatGL predictions to filter the set, then run expensive matcalc calculations only on the top fraction.
3. **Present a plan** — Before executing, outline the approach to the user (how many candidates, which properties, estimated runtime) so they can adjust scope.
4. **Iterative refinement** — Cached properties don't need re-computation if criteria change; you can re-rank without re-running predictions.

**Batch scripts:** Use `matclaw_sdk` (`pip install -e /path/to/MatClaw/sdk/`). Import tools directly: `from matclaw_sdk import tool_name`. See [examples/batch_screening_example.py](examples/batch_screening_example.py).

**For full execution patterns, failure handling, and checkpoint implementation, see [references/execution-guide.md](references/execution-guide.md)**

---

## Integration

- **Input:** From `candidate-generator` or any source of CIF structures
- **Output:** Ranked candidates with properties → `synthesis-planner` or DFT workflow
- **Stability analysis:** For `energy_above_hull` or custom hull construction, consider the `stability-analyzer` skill

---

## Reference files

| File | Contents | When to read |
|------|----------|-------------|
| `references/preprocessing-guide.md` | Detailed disorder handling, ordering decision trees, physical justification | When processing disordered inputs |
| `references/execution-guide.md` | Workflow patterns, failure recovery, checkpoint implementation, batch execution | When implementing end-to-end screening or working with >20 candidates |
| `references/ml-calculations-guide.md` | MatGL vs matcalc selection, model recommendations, parameter tuning | When choosing specific ML tools or configuring matcalc parameters |
| `examples/batch_screening_example.py` | Complete reference script with hierarchical retrieval, checkpointing, ranking | When building custom batch screening scripts |

---

**Last updated:** 2025-06-10  
**Skill version:** 3.0 (refactored with guiding principles)
