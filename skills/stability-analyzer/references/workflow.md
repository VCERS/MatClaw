# Stability Analyzer Workflow Reference

This reference is for the model using the skill during actual execution. Read it when the task requires making the route decision or assembling the final answer shape.

## Route Decision Checklist

### Route A: MP-backed lookup

Choose this route if the request is satisfied by known Materials Project thermodynamic data.

Strong signals:
- formula-only request
- user asks for a quick stability answer
- user explicitly wants the known Materials Project result
- candidate-screener Phase 1 sanity check

Typical actions:
1. Search or identify the likely MP entry.
2. Retrieve thermodynamic properties.
3. Report `formation_energy_per_atom`, `energy_above_hull`, and `is_stable`.
4. Include polymorph context if there are multiple candidate matches.

Recommended tools:
- `mp_search_materials`
- `mp_get_material_properties`

### Route B: Custom self-consistent hull

Choose this route if the answer depends on the specific structure or a custom energy workflow.

Strong signals:
- explicit CIF or POSCAR input
- user asks for a novel-material or structure-specific answer
- disorder or doping is present
- user requests self-consistent MLIP hull analysis
- candidate-screener Phase 2 computed `energy_above_hull`

Typical actions:
1. Resolve disorder if needed.
2. Assemble a competitor set in the target chemical system.
3. Reevaluate target, competitors, and unary elemental references with one shared energetics workflow.
4. Build the self-consistent hull.
5. Report `energy_above_hull` with decomposition and caveats.

Recommended tools:
- `pymatgen_majority_orderer`
- `pymatgen_enumeration_orderer`
- `pymatgen_sqs_orderer`
- `mp_search_materials`
- `mp_get_material_properties`
- `matcalc_calc_energetics`

Bundled helper:
- Use [../scripts/build_self_consistent_hull.py](../scripts/build_self_consistent_hull.py) for deterministic hull construction from reevaluated energies.

### Custom hull execution checklist

1. Decide whether disorder resolution is required.
2. If disordered, choose majority, enumeration, or SQS ordering.
3. Query the target chemical system and unary elements for a competitor set.
4. Retain stable and relevant near-hull competitors.
5. Reevaluate target, competitors, and elemental references with one shared energetics workflow.
6. Build the self-consistent hull from reevaluated totals.
7. Report `energy_above_hull`, decomposition, and caveats.

### JSON handoff format for the helper

The bundled helper expects a JSON file shaped like this:

```json
{
	"target": {
		"formula": "SrNb2O6",
		"composition": {"Sr": 1, "Nb": 2, "O": 6},
		"total_energy_per_atom": -3.024986,
		"total_energy_eV": -108.899498,
		"num_atoms": 36
	},
	"competitors": [
		{
			"formula": "Sr",
			"composition": {"Sr": 1},
			"total_energy_per_atom": 0.0,
			"total_energy_eV": 0.0,
			"num_atoms": 1,
			"material_id": "mp-139"
		}
	],
	"hull_tolerance": 0.07
}
```

`competitors` must include unary elemental entries for every element in the target.

## Candidate-Screener Mapping

### Phase 1
- use only the MP-backed lookup route
- treat this as a cheap known-data sanity check
- do not launch custom hull computation here

### Phase 2
- use the custom hull route when `energy_above_hull` is part of the computed screening criteria
- keep the route explicit in the report so the user can distinguish lookup-based and computed stability results

## Output Expectations

Always begin the final summary with:

```text
Stability route: MP-backed lookup | custom self-consistent hull
```

### For MP-backed lookup
Include:
- formula
- MP material ID if available
- `formation_energy_per_atom`
- `energy_above_hull`
- `is_stable`
- short note on polymorphs or decomposition if relevant

### For custom hull
Include:
- formula
- ordering strategy if any
- calculator/settings if relevant
- `formation_energy_per_atom`
- `energy_above_hull`
- stability level
- decomposition products
- explicit caveat about custom hull completeness and approximation quality

Numerical edge case guidance:
- treat `|energy_above_hull| < 1e-6` as numerical zero
- if the helper has to clamp a tiny negative value to zero, say that the target is effectively on the hull
- do not overinterpret floating-point sign artifacts

## Failure Handling

If the route is ambiguous:
- prefer MP-backed lookup for cheap known-data requests
- prefer custom hull for explicit-structure requests

If MP lookup finds no convincing match:
- say so explicitly
- only escalate to custom hull if the user’s request actually requires it

If the custom hull route is too expensive for the current workflow stage:
- report that it belongs in Phase 2 or a dedicated stability pass
- do not silently replace it with a weaker heuristic