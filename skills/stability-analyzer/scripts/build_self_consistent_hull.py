"""Build a self-consistent formation-energy hull from reevaluated MLIP total energies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition


def _composition_from_dict(data: dict) -> Composition:
    return Composition(data)


def _formation_energy_per_atom(composition: Composition, total_energy_per_atom: float, refs: dict[str, float]) -> float:
    reference_energy = 0.0
    for element, fraction in composition.fractional_composition.as_dict().items():
        reference_energy += fraction * refs[element]
    return total_energy_per_atom - reference_energy


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python build_self_consistent_hull.py <input.json>", file=sys.stderr)
        return 2

    payload = json.loads(Path(sys.argv[1]).read_text())
    target = payload["target"]
    competitors = payload["competitors"]
    hull_tolerance = float(payload.get("hull_tolerance", 0.07))

    target_comp = _composition_from_dict(target["composition"])
    target_elements = {str(el) for el in target_comp.elements}

    elemental_reference_energies: dict[str, float] = {}
    competitor_rows = []
    for row in competitors:
        composition = _composition_from_dict(row["composition"])
        total_energy_per_atom = float(row["total_energy_per_atom"])
        total_energy_eV = float(row["total_energy_eV"])
        entry = {
            "formula": row["formula"],
            "material_id": row.get("material_id"),
            "composition": composition,
            "total_energy_per_atom": total_energy_per_atom,
            "total_energy_eV": total_energy_eV,
            "num_atoms": int(row["num_atoms"]),
        }
        competitor_rows.append(entry)

        if len(composition.elements) == 1:
            element = str(composition.elements[0])
            current = elemental_reference_energies.get(element)
            if current is None or total_energy_per_atom < current:
                elemental_reference_energies[element] = total_energy_per_atom

    missing = sorted(target_elements - set(elemental_reference_energies))
    if missing:
        raise ValueError(f"Missing elemental reference energies for {missing}")

    hull_entries = []
    output_competitors = []
    for row in competitor_rows:
        formation_energy_per_atom = _formation_energy_per_atom(
            row["composition"],
            row["total_energy_per_atom"],
            elemental_reference_energies,
        )
        hull_entries.append(
            PDEntry(row["composition"], formation_energy_per_atom * row["num_atoms"])
        )
        output_competitors.append(
            {
                "formula": row["formula"],
                "material_id": row["material_id"],
                "total_energy_per_atom": round(row["total_energy_per_atom"], 6),
                "formation_energy_per_atom": round(formation_energy_per_atom, 6),
            }
        )

    target_formation_energy_per_atom = _formation_energy_per_atom(
        target_comp,
        float(target["total_energy_per_atom"]),
        elemental_reference_energies,
    )
    target_entry = PDEntry(target_comp, target_formation_energy_per_atom * int(target["num_atoms"]))

    pd = PhaseDiagram(hull_entries)
    try:
        decomp, e_above_hull = pd.get_decomp_and_e_above_hull(target_entry)
    except ValueError as exc:
        hull_energy_per_atom = pd.get_hull_energy_per_atom(target_comp)
        e_above_hull = target_formation_energy_per_atom - hull_energy_per_atom
        if e_above_hull <= 1e-6:
            decomp = {target_entry: 1.0}
            e_above_hull = 0.0
        else:
            raise ValueError(
                f"Failed to evaluate target entry against the custom hull: {exc}"
            ) from exc

    if abs(e_above_hull) < 1e-6:
        e_above_hull = 0.0

    if e_above_hull <= hull_tolerance:
        stability_level = "stable"
        is_stable = True
    elif e_above_hull <= 0.1:
        stability_level = "metastable"
        is_stable = False
    else:
        stability_level = "unstable"
        is_stable = False

    result = {
        "composition": target_comp.reduced_formula,
        "energy_above_hull": round(float(e_above_hull), 6),
        "is_stable": is_stable,
        "stability_level": stability_level,
        "target_energy_info": {
            "formation_energy_per_atom": round(target_formation_energy_per_atom, 6),
            "total_energy_per_atom": round(float(target["total_energy_per_atom"]), 6),
            "total_energy_eV": round(float(target["total_energy_eV"]), 6),
        },
        "elemental_reference_energies": {
            key: round(val, 6) for key, val in sorted(elemental_reference_energies.items())
        },
        "decomposition": {
            "decomposition_products": [phase.composition.reduced_formula for phase in decomp],
            "product_fractions": {
                phase.composition.reduced_formula: round(float(frac), 6)
                for phase, frac in decomp.items()
            },
        },
        "competitors": output_competitors,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())