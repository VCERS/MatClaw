---
name: stability-analyzer
description: >
  Analyze thermodynamic stability of inorganic materials by routing between two workflows: a cheap Materials Project-backed lookup path for known compositions, and a custom self-consistent MLIP hull workflow for novel or structure-specific materials. Use this skill whenever the user asks whether a material is stable, requests energy above hull, decomposition products, polymorph context, or wants to include stability as a screening criterion. This skill is intended to become the single orchestration layer for stability analysis, with workflow branching handled in the skill rather than inside an MCP tool.
---

# Stability Analyzer Skill

It unifies two stability workflows under one routing layer:
1. **MP-backed lookup** for known compositions and existing Materials Project entries
2. **Custom hull analysis** for novel, disordered, doped, or structure-specific materials

---

## Core Principle

Stability analysis is a workflow decision, not a single tool call.

The key distinction is not whether the user says “stability” or “energy above hull,” but whether the answer should come from:
- existing Materials Project thermodynamic data
- or a freshly constructed, self-consistent custom hull

This skill should own that routing logic.

---

## When To Use

Use this skill when the user asks any of the following:
- "Is this material thermodynamically stable?"
- "What is the energy above hull of this composition or structure?"
- "Will this material decompose, and into what?"
- "Is this candidate likely synthesizable from a thermodynamic perspective?"
- "Use stability as a screening criterion"
- "Compare stable and unstable candidates"

This skill should apply both to:
- simple known formulas like `SrNb2O6`
- explicit CIF or POSCAR structures
- disordered or doped materials needing ordering first

---

## Routing Logic

### Route A: MP-backed lookup path

Use this route when all of the following are true:
- the user’s request can be answered compositionally rather than structure-specifically
- the material is likely represented in Materials Project
- no self-consistent recomputation is requested
- no externally computed energies need to be compared against MP entries

Typical triggers:
- formula-only requests
- “just tell me whether this known material is stable in MP”
- quick screening sanity checks

Preferred tools for this route:
- `mp_search_materials`
- `mp_get_material_properties`

Expected outputs:
- MP material IDs, if available
- `formation_energy_per_atom`
- `energy_above_hull`
- `is_stable`
- polymorph context if relevant

### Route B: Custom hull path

Use this route when any of the following are true:
- the user provides an explicit structure and wants a structure-specific answer
- the structure is novel, hypothetical, disordered, doped, or otherwise not a simple known MP lookup
- the user explicitly requests a self-consistent MLIP hull
- the request involves externally computed structures or energies
- the candidate-screener workflow needs `energy_above_hull` as a Phase 2 computed property

Preferred tools and helpers for this route:
- `pymatgen_majority_orderer`
- `pymatgen_enumeration_orderer`
- `pymatgen_sqs_orderer`
- `mp_search_materials`
- `mp_get_material_properties`
- `matcalc_calc_energetics`
- `scripts/build_self_consistent_hull.py`

Expected outputs:
- self-consistent `energy_above_hull`
- decomposition products
- elemental reference energies
- ordering strategy used, if any
- caveats about approximation quality

Detailed custom-hull execution guidance lives below and in [references/workflow.md](references/workflow.md).

---

## Tool Responsibilities

This skill assumes the following separation of concerns:

### Primitive tools
- Materials Project tools retrieve data
- ordering tools convert disordered structures into ordered approximations
- energetics tools evaluate structures on a common MLIP energy surface

### Skill layer
- decides lookup vs custom-hull route
- decides whether disorder resolution is required
- decides what competitor set is needed
- decides how to present the result as a stability answer

The skill should avoid hiding these decisions. It should make the route explicit in the final response.

---

## Proposed Workflow

### Step 1: Normalize the input

Parse whether the user gave:
- a formula only
- a CIF or POSCAR structure
- a disordered structure
- a known MP material identifier

### Step 2: Decide route

Use this decision rule:

```text
If the answer can be satisfied by known MP thermodynamic data,
take the MP-backed lookup path.

If the user requests a structure-specific or novel-material answer,
take the custom hull path.
```

### Step 3A: MP-backed lookup path

1. Search MP by formula or chemical system.
2. If a known material is found, retrieve thermodynamic data.
3. Return a stability summary from MP-backed values.
4. If relevant, show polymorphs or multiple MP matches.

### Step 3B: Custom hull path

1. Resolve disorder if needed using orderer tools.
2. Discover a competitor set in the target chemical system.
3. Reevaluate target, competitors, and elemental references with one shared energetics workflow.
4. Build the self-consistent hull.
5. Return a stability summary with decomposition and caveats.

---

## Custom Hull Execution Details

Use this section when the skill chooses the custom self-consistent hull route.

### Disorder resolution

If the target structure has fractional occupancies, choose an ordering strategy before any energetics:
- dilute dopants or quick screening: `pymatgen_majority_orderer`
- small-cell site-specific study: `pymatgen_enumeration_orderer`
- concentrated disorder or solid solutions: `pymatgen_sqs_orderer`

Default reasoning:
- `< 10%` dopant concentration: majority ordering is an acceptable first pass
- `10-20%`: majority ordering for screening, but call out the approximation
- `> 20%`: prefer SQS or enumeration

Always report the ordering strategy in the final summary.

### Competitor discovery

Use `mp_search_materials` over the full chemical system and also query unary element systems explicitly. Keep:
- stable entries
- near-hull metastable entries that could matter after reevaluation
- unary elemental reference structures for every element in the target composition

Then use `mp_get_material_properties(..., properties=["structure", "thermodynamic", "basic"])` to fetch the competitor structures.

Be explicit that this is still a curated competitor set, not a claim of exhaustive phase-space completeness.

### Shared energetics workflow

Call `matcalc_calc_energetics` on:
- the target
- every retained competitor
- every unary elemental reference entry

Use the same settings for all calls:
- `calculator`
- `elemental_refs`
- `relax_structure`
- `fmax`
- `optimizer`
- `use_gs_reference`

The point is to place all entries on one internally consistent potential-energy surface.

### Hull construction helper

Create a JSON file with:
- target composition and total energy
- competitor compositions and total energies
- unary elemental reference entries
- `hull_tolerance`

Then run:

```bash
python skills/stability-analyzer/scripts/build_self_consistent_hull.py <input.json>
```

The helper will:
- derive elemental reference energies from reevaluated unary phases
- convert totals to self-consistent formation energies
- construct the convex hull with pymatgen
- emit `energy_above_hull`, decomposition, and per-entry formation energies

---

## Candidate-Screener Integration

This skill is the intended home for all stability-specific decision logic in candidate screening.

Recommended split:
- **Phase 1:** Use the MP-backed route only for cheap known-data sanity checks.
- **Phase 2:** Use the custom hull route when `energy_above_hull` is a computed screening property.

This keeps the screening workflow aligned with computational cost while still treating “stability” as one conceptual property family.

---

## Output Contract

Every response should declare which route was taken:

```text
Stability route: MP-backed lookup | custom self-consistent hull
```

For the MP-backed route, include:
- formula
- MP material ID, if known
- `formation_energy_per_atom`
- `energy_above_hull`
- `is_stable`
- polymorph or decomposition notes if useful

For the custom-hull route, include:
- formula
- ordering strategy, if any
- calculator used
- `formation_energy_per_atom`
- `energy_above_hull`
- stability level
- decomposition products
- explicit caveats about custom hull completeness and approximation quality

Preferred custom-hull summary shape:

```text
Target: <formula>
Ordering strategy: <none|majority|enumeration|sqs>
Calculator: <calculator>
Relaxed: <true|false>
Formation energy per atom: <value> eV/atom
Energy above hull: <value> eV/atom
Stability level: <stable|metastable|unstable>
Decomposition: <products or self>
Notes: <important caveats>
```

---

## Bundled Resources

- Read [references/workflow.md](references/workflow.md) for the operational routing checklist and per-route output expectations.
- Read [references/migration-notes.md](references/migration-notes.md) for the merged-skill architecture notes and cleanup intent.
- Use [scripts/build_self_consistent_hull.py](scripts/build_self_consistent_hull.py) for deterministic hull construction from reevaluated energies.
- Use [evals/evals.json](evals/evals.json) as the initial prompt set for validating that the skill chooses the correct route.

---