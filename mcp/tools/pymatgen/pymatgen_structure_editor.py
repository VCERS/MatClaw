"""
Tool for applying explicit site-level edits to crystal structures.

Supports direct structure manipulation operations such as removing specific sites,
replacing selected sites, and inserting new atoms at explicit coordinates.
This is intended for constructing user-chosen multi-defect or site-specific models
that are not naturally expressed as species-wide enumeration tasks.
"""

from typing import Dict, Any, Optional, List, Union, Annotated
from pydantic import Field


def pymatgen_structure_editor(
    input_structures: Annotated[
        Union[str, List[str]],
        Field(
            description=(
                "Input structure(s) to edit. Can be a single CIF/POSCAR string or a list of "
                "CIF/POSCAR strings. All operations are applied in sequence to each input structure."
            )
        )
    ],
    operations: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "Ordered list of edit operations to apply. Supported operations: "
                "remove_sites, replace_sites, insert_sites. "
                "Selection modes for remove/replace: index, nearest_to_coords."
            )
        )
    ],
    selection_tolerance: Annotated[
        float,
        Field(
            default=0.2,
            ge=0.001,
            le=5.0,
            description=(
                "Maximum matching distance in Angstroms when selection mode is nearest_to_coords. "
                "Default: 0.2 Å."
            )
        )
    ] = 0.2,
    preserve_site_properties: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True (default), preserve site properties when replacing a site. "
                "Inserted sites may optionally provide their own properties."
            )
        )
    ] = True,
    validate_structure: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True (default), validate the edited structure after all operations by checking "
                "that no pair of atoms is closer than min_distance."
            )
        )
    ] = True,
    min_distance: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.1,
            le=3.0,
            description=(
                "Minimum allowed interatomic distance in Angstroms during validation. "
                "Default: 0.5 Å."
            )
        )
    ] = 0.5,
    output_format: Annotated[
        str,
        Field(
            default="cif",
            description=(
                "Output format for returned structures. "
                "'cif' (default), 'poscar', or 'json'."
            )
        )
    ] = "cif"
) -> Dict[str, Any]:
    """
    Apply explicit site-level edits to one or more structures.

    This tool is intentionally low-level and deterministic. Unlike species-level
    generators, it does not enumerate symmetry-inequivalent defects. Instead, it
    applies user-specified operations to user-specified sites in sequence.

    Supported operations
    --------------------
    remove_sites:
        {
            "type": "remove_sites",
            "selection": {"mode": "index", "indices": [1, 2]},
            "label": "divacancy"
        }

    replace_sites:
        {
            "type": "replace_sites",
            "selection": {"mode": "nearest_to_coords", "coords": [[0, 0, 0]]},
            "new_species": "Na"
        }

    insert_sites:
        {
            "type": "insert_sites",
            "sites": [
                {
                    "species": "Li",
                    "coords": [0.25, 0.25, 0.25],
                    "coords_are_fractional": True
                }
            ]
        }

    Returns
    -------
    dict:
        success         (bool)
        count           (int) number of edited structures returned
        structures      (list) structures in requested output_format
        metadata        (list) per-structure edit metadata
        input_info      (dict) summary of input structures
        edit_summary    (dict) summary of requested operations
        message         (str)
        warnings        (list, optional)
        error           (str, optional)
    """
    import json
    import numpy as np

    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        from pymatgen.io.vasp import Poscar
    except ImportError as e:
        return {
            "success": False,
            "error": f"Failed to import pymatgen: {str(e)}. Install with: pip install pymatgen"
        }

    valid_formats = {"poscar", "cif", "json"}
    if output_format not in valid_formats:
        return {
            "success": False,
            "error": f"Invalid output_format '{output_format}'. Must be one of {sorted(valid_formats)}."
        }

    if isinstance(input_structures, str):
        raw_inputs = [input_structures]
    elif isinstance(input_structures, list) and all(isinstance(item, str) for item in input_structures):
        raw_inputs = input_structures
    else:
        return {
            "success": False,
            "error": f"Invalid input_structures type: {type(input_structures).__name__}"
        }

    if not operations or not isinstance(operations, list):
        return {"success": False, "error": "operations must be a non-empty list."}

    structures = []
    for i, item in enumerate(raw_inputs):
        try:
            if "data_" in item or "_cell_length" in item:
                struct = Structure.from_str(item, fmt="cif")
            else:
                struct = Structure.from_str(item, fmt="poscar")
            structures.append(struct)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse input structure {i}: {str(e)}"}

    def _format_structure(struct: Structure) -> str:
        if output_format == "cif":
            return str(CifWriter(struct))
        if output_format == "poscar":
            return str(Poscar(struct))
        return json.dumps(struct.as_dict())

    def _species_matches(site_symbol: str, species_constraint: Optional[Union[str, List[str]]], position: int) -> bool:
        if species_constraint is None:
            return True
        if isinstance(species_constraint, str):
            return site_symbol == species_constraint
        if len(species_constraint) == 1:
            return site_symbol == species_constraint[0]
        if len(species_constraint) == position + 1:
            return site_symbol == species_constraint[position]
        return site_symbol in set(species_constraint)

    def _select_sites(
        struct: Structure,
        site_records: List[Dict[str, Any]],
        selection: Dict[str, Any],
        tolerance: float,
    ) -> List[int]:
        if not isinstance(selection, dict):
            raise ValueError("selection must be a dictionary.")

        mode = selection.get("mode")
        species_constraint = selection.get("species")

        if mode == "index":
            indices = selection.get("indices")
            if not isinstance(indices, list) or not indices:
                raise ValueError("selection.indices must be a non-empty list for mode='index'.")
            selected = []
            for pos, idx in enumerate(indices):
                if not isinstance(idx, int):
                    raise ValueError("selection.indices must contain integers only.")
                if idx < 0 or idx >= len(struct):
                    raise ValueError(f"Site index {idx} is out of range for structure with {len(struct)} sites.")
                symbol = struct[idx].specie.symbol
                if not _species_matches(symbol, species_constraint, pos):
                    raise ValueError(
                        f"Site index {idx} has species '{symbol}', which does not match the requested species constraint."
                    )
                selected.append(idx)
        elif mode == "nearest_to_coords":
            coords = selection.get("coords")
            coords_are_fractional = bool(selection.get("coords_are_fractional", False))
            if not isinstance(coords, list) or not coords:
                raise ValueError("selection.coords must be a non-empty list for mode='nearest_to_coords'.")

            selected = []
            for pos, coord in enumerate(coords):
                if not (isinstance(coord, list) and len(coord) == 3):
                    raise ValueError("Each coordinate must be a 3-element list.")

                frac_target = np.array(coord, dtype=float)
                if not coords_are_fractional:
                    frac_target = struct.lattice.get_fractional_coords(frac_target)

                best_idx = None
                best_dist = float("inf")
                for idx, site in enumerate(struct):
                    symbol = site.specie.symbol
                    if not _species_matches(symbol, species_constraint, pos):
                        continue
                    dist = float(struct.lattice.get_distance_and_image(frac_target, site.frac_coords)[0])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx is None:
                    raise ValueError("No matching site found for nearest_to_coords selection.")
                if best_dist > tolerance:
                    raise ValueError(
                        f"Nearest matching site is {best_dist:.3f} Å away, exceeding selection_tolerance={tolerance:.3f} Å."
                    )
                selected.append(best_idx)
        else:
            raise ValueError(
                f"Unsupported selection mode '{mode}'. Supported modes: 'index', 'nearest_to_coords'."
            )

        if len(selected) != len(set(selected)):
            raise ValueError("Selection resolved to duplicate site indices within one operation.")

        return selected

    def _validate_min_distance(struct: Structure, threshold: float) -> None:
        if len(struct) < 2:
            return
        dm = np.array(struct.distance_matrix, dtype=float)
        dm += np.eye(len(struct)) * 1e9
        min_found = float(np.min(dm))
        if min_found < threshold:
            raise ValueError(
                f"Edited structure violates min_distance={threshold:.3f} Å; closest pair is {min_found:.3f} Å."
            )

    generated_structures: List[str] = []
    metadata_list: List[Dict[str, Any]] = []
    warnings: List[str] = []

    edit_summary = {
        "n_operations": len(operations),
        "operations": [
            {
                "index": i + 1,
                "type": op.get("type"),
                "label": op.get("label")
            }
            for i, op in enumerate(operations)
        ],
    }

    for struct_idx, struct in enumerate(structures):
        working = struct.copy()
        site_records = [
            {
                "original_index": i,
                "original_species": site.specie.symbol,
                "original_frac_coords": list(site.frac_coords),
                "original_cart_coords": list(site.coords),
            }
            for i, site in enumerate(struct)
        ]
        operation_log = []

        try:
            for op_idx, operation in enumerate(operations):
                if not isinstance(operation, dict):
                    raise ValueError(f"Operation {op_idx + 1} must be a dictionary.")
                op_name = operation.get("type")
                label = operation.get("label")

                if op_name == "remove_sites":
                    selected = _select_sites(working, site_records, operation.get("selection"), selection_tolerance)
                    removed_sites = []
                    for idx in sorted(selected, reverse=True):
                        site = working[idx]
                        record = site_records[idx]
                        removed_sites.append({
                            "site_index_current": idx,
                            "site_index_original": record["original_index"],
                            "species": site.specie.symbol,
                            "frac_coords": list(site.frac_coords),
                            "cart_coords": list(site.coords),
                        })
                        working.remove_sites([idx])
                        site_records.pop(idx)
                    removed_sites.reverse()
                    operation_log.append({
                        "type": op_name,
                        "label": label,
                        "n_sites_removed": len(removed_sites),
                        "sites_removed": removed_sites,
                    })

                elif op_name == "replace_sites":
                    new_species = operation.get("new_species")
                    if not isinstance(new_species, str) or not new_species:
                        raise ValueError("replace_sites requires a non-empty 'new_species' string.")
                    selected = _select_sites(working, site_records, operation.get("selection"), selection_tolerance)
                    replaced_sites = []
                    for idx in selected:
                        site = working[idx]
                        record = site_records[idx]
                        old_species = site.specie.symbol
                        properties = site.properties if preserve_site_properties else None
                        working.replace(idx, new_species, properties=properties)
                        replaced_sites.append({
                            "site_index_current": idx,
                            "site_index_original": record["original_index"],
                            "old_species": old_species,
                            "new_species": new_species,
                            "frac_coords": list(working[idx].frac_coords),
                            "cart_coords": list(working[idx].coords),
                        })
                    operation_log.append({
                        "type": op_name,
                        "label": label,
                        "n_sites_replaced": len(replaced_sites),
                        "sites_replaced": replaced_sites,
                    })

                elif op_name == "insert_sites":
                    sites = operation.get("sites")
                    if not isinstance(sites, list) or not sites:
                        raise ValueError("insert_sites requires 'sites' as a non-empty list.")

                    inserted_sites = []
                    for site_spec in sites:
                        if not isinstance(site_spec, dict):
                            raise ValueError("Each insert_sites entry must be a dictionary.")

                        species = site_spec.get("species")
                        coords = site_spec.get("coords")
                        coords_are_fractional = bool(site_spec.get("coords_are_fractional", False))
                        properties = site_spec.get("properties")
                        if not isinstance(species, str) or not species:
                            raise ValueError("Each insert_sites entry requires a non-empty 'species' string.")
                        if not (isinstance(coords, list) and len(coords) == 3):
                            raise ValueError("Each insert_sites entry requires 'coords' as a 3-element list.")

                        working.append(
                            species,
                            coords,
                            coords_are_cartesian=not coords_are_fractional,
                            properties=properties,
                        )
                        inserted_idx = len(working) - 1
                        inserted_site = working[inserted_idx]
                        site_records.append({
                            "original_index": None,
                            "original_species": None,
                            "original_frac_coords": None,
                            "original_cart_coords": None,
                        })
                        inserted_sites.append({
                            "site_index_current": inserted_idx,
                            "site_index_original": None,
                            "species": species,
                            "frac_coords": list(inserted_site.frac_coords),
                            "cart_coords": list(inserted_site.coords),
                        })

                    operation_log.append({
                        "type": op_name,
                        "label": label,
                        "n_sites_inserted": len(inserted_sites),
                        "sites_inserted": inserted_sites,
                    })
                else:
                    raise ValueError(
                        f"Unsupported operation '{op_name}'. Supported operations: remove_sites, replace_sites, insert_sites."
                    )

            if validate_structure:
                _validate_min_distance(working, min_distance)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed while editing structure {struct_idx}: {str(e)}",
                "warnings": warnings,
            }

        generated_structures.append(_format_structure(working))
        metadata_list.append({
            "index": len(generated_structures),
            "formula": working.composition.reduced_formula,
            "composition": str(working.composition),
            "n_sites": len(working),
            "volume": round(float(working.volume), 6),
            "structure_index": struct_idx,
            "operations_applied": operation_log,
        })

    if not generated_structures:
        return {
            "success": False,
            "error": "No edited structures were generated.",
            "warnings": warnings,
        }

    return {
        "success": True,
        "count": len(generated_structures),
        "structures": generated_structures,
        "metadata": metadata_list,
        "input_info": {
            "n_input_structures": len(structures),
            "input_formulas": [s.composition.reduced_formula for s in structures],
            "n_sites": [len(s) for s in structures],
        },
        "edit_summary": edit_summary,
        "message": f"Applied {len(operations)} operation(s) to {len(generated_structures)} structure(s).",
        "warnings": warnings,
    }