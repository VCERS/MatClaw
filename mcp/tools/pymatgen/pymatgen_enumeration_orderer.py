"""
Generate multiple ordered candidates from a disordered structure within a chosen supercell.

This tool starts from a structure with partial occupancies, builds a supercell, and uses
pymatgen's ordering transformation to produce one or more fully ordered candidates. The
returned candidates can be ranked by electrostatic energy or by size.

Important scope note:
    This tool explores orderings inside the user-selected supercell. It is useful for
    generating several plausible ordered configurations, but it should not be described as a
    complete search over every possible supercell choice.

Best for:
    - Site-specific dopant studies
    - Small to moderate ordering problems where several configurations should be tested
    - Building DFT candidate pools for partially occupied inputs

Relationship to the other orderers:
    - pymatgen_majority_orderer: one compact approximation, no supercell
    - pymatgen_enumeration_orderer: multiple ordered candidates within a chosen supercell
    - pymatgen_sqs_orderer: quasirandom ordered supercells for random-alloy behaviour
"""

from typing import Dict, Any, Optional, List, Union, Annotated
from pydantic import Field


def pymatgen_enumeration_orderer(
    input_structures: Annotated[
        Union[str, List[str]],
        Field(
            description=(
                "Input structure(s) with fractional site occupancies (disordered). "
                "Accepts CIF string or list of CIF strings. "
                "Each structure must have at least one site with partial occupancy; "
                "fully ordered structures are skipped unless check_ordered_input=False."
            )
        )
    ],
    supercell_size: Annotated[
        int,
        Field(
            default=2,
            ge=1,
            le=4,
            description=(
                "Supercell size multiplier (1–4). "
                "Creates a supercell of size [supercell_size, supercell_size, 1] "
                "to accommodate fractional occupancies before ordering. "
                "Larger values allow more ordering possibilities but increase computation time. "
                "Default: 2."
            )
        )
    ] = 2,
    n_structures: Annotated[
        int,
        Field(
            default=20,
            ge=1,
            le=500,
            description=(
                "Maximum number of ordered structures to return per input structure (1–500). "
                "The enumeration may find fewer configurations than this limit. "
                "Default: 20."
            )
        )
    ] = 20,
    sort_by: Annotated[
        str,
        Field(
            default="ewald",
            description=(
                "Ranking criterion for returned structures. "
                "'ewald': rank by Ewald electrostatic energy — lowest energy first. "
                "  Requires oxidation states; use add_oxidation_states=True if not decorated. "
                "  Best criterion for ionic materials (oxides, fluorides, etc.). "
                "'num_sites': rank by supercell size — smallest supercells first. "
                "'random': return in arbitrary order (no re-ranking). "
                "Default: 'ewald'."
            )
        )
    ] = "ewald",
    symm_prec: Annotated[
        float,
        Field(
            default=0.1,
            ge=0.001,
            le=0.5,
            description=(
                "Symmetry tolerance in Angstroms for identifying equivalent configurations (0.001–0.5). "
                "Higher values merge more structures as equivalent (fewer results). "
                "Lower values distinguish more subtle symmetry differences (more results). "
                "Default: 0.1 Å."
            )
        )
    ] = 0.1,
    refine_structure: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True, re-symmetrizes the input structure using SpacegroupAnalyzer "
                "before passing it to the enumerator. Recommended to ensure the symmetry "
                "operations used during enumeration are correct. Default: True."
            )
        )
    ] = True,
    check_ordered_input: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True (default), structures that are already fully ordered (no partial "
                "occupancies) are skipped and a warning is emitted. "
                "If False, ordered structures are passed to the enumerator anyway "
                "(useful when you want to systematically generate supercell orderings of an "
                "already-ordered phase for defect or substitution studies)."
            )
        )
    ] = True,
    add_oxidation_states: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True (default) and sort_by='ewald', automatically assigns oxidation "
                "states to the structure using pymatgen's BVAnalyzer before enumeration. "
                "This is required for Ewald energy ranking. "
                "If BVAnalyzer fails, the tool falls back to sort_by='num_sites' and "
                "records a warning. Set to False if the structure already carries oxidation "
                "states or if you want to suppress automatic decoration."
            )
        )
    ] = True,
    output_format: Annotated[
        str,
        Field(
            default="cif",
            description=(
                "Output format for the returned structures. "
                "'cif': CIF string (default). "
                "'poscar': VASP POSCAR string. "
                "'json': JSON-serialised Structure dict string."
            )
        )
    ] = "cif"
) -> Dict[str, Any]:
    """
    Generate ordered candidates from disordered structures with partial occupancies.

    The workflow is:
    1. Parse one or more disordered input structures.
    2. Expand each structure into the requested supercell.
    3. Use pymatgen's ordering transformation to generate fully ordered candidates.
    4. Rank the resulting candidates by Ewald energy, number of sites, or random order.

    The returned structures are fully ordered and suitable for downstream relaxation,
    property prediction, or DFT setup.

    Returns:
        dict:
            success             (bool)  Whether at least one ordered candidate was generated.
            count               (int)   Total number of returned ordered structures.
            structures          (list)  Ordered structures in the requested output_format.
            metadata            (list)  Per-structure metadata:
                index               (int)   Sequential index (1-based).
                source_structure    (str)   Reduced formula of the input structure.
                formula             (str)   Reduced formula of this ordered structure.
                n_sites             (int)   Number of sites in the returned structure.
                supercell_size      (int)   Approximate size multiplier relative to the parent cell.
                volume              (float) Cell volume in Å³.
                space_group_number  (int)   Space group number (if determinable).
                space_group_symbol  (str)   Hermann-Mauguin symbol (if determinable).
                ewald_energy        (float) Ewald energy in eV when available.
                is_ordered          (bool)  True for valid returned candidates.
            input_info          (dict)  Summary of the input structures.
            enumeration_params  (dict)  Parameters used for the run.
            message             (str)   Human-readable summary.
            warnings            (list)  Any non-fatal warnings generated.
            error               (str)   Error message if success=False.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as e:
        return {
            "success": False,
            "error": f"Failed to import pymatgen: {e}. Install with: pip install pymatgen"
        }

    # Validate parameters
    valid_formats = {"poscar", "cif", "json"}
    if output_format not in valid_formats:
        return {
            "success": False,
            "error": f"Invalid output_format '{output_format}'. Must be one of {sorted(valid_formats)}."
        }

    valid_sort = {"ewald", "num_sites", "random"}
    if sort_by not in valid_sort:
        return {
            "success": False,
            "error": f"Invalid sort_by '{sort_by}'. Must be one of {sorted(valid_sort)}."
        }



    # Parse input structures
    if isinstance(input_structures, str):
        raw_list = [input_structures]
    elif isinstance(input_structures, list):
        raw_list = input_structures
    else:
        return {
            "success": False,
            "error": f"Invalid input_structures type: {type(input_structures).__name__}."
        }

    structures: List[Structure] = []
    for i, item in enumerate(raw_list):
        try:
            if isinstance(item, str):
                structures.append(Structure.from_str(item, fmt="cif"))
            else:
                return {
                    "success": False,
                    "error": (
                        f"Input structure {i} must be a CIF string, "
                        f"got {type(item).__name__}."
                    )
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to parse input structure {i}: {e}"}

    if not structures:
        return {"success": False, "error": "No valid input structures provided."}

    # Import OrderDisorderedStructureTransformation
    try:
        from pymatgen.transformations.standard_transformations import (
            OrderDisorderedStructureTransformation,
        )
    except ImportError as e:
        return {
            "success": False,
            "error": (
                f"Failed to import OrderDisorderedStructureTransformation: {e}. "
                "Ensure pymatgen is installed: pip install pymatgen"
            )
        }

    # Main enumeration loop
    generated_structures: List[Any] = []
    metadata_list: List[Dict[str, Any]] = []
    warnings: List[str] = []
    skipped_ordered: List[str] = []

    for struct in structures:
        src_formula = struct.composition.reduced_formula

        # Skip already-ordered structures if requested
        if check_ordered_input and struct.is_ordered:
            skipped_ordered.append(src_formula)
            warnings.append(
                f"Structure '{src_formula}' is already fully ordered (no partial occupancies) "
                "and was skipped. Set check_ordered_input=False to enumerate it anyway."
            )
            continue

        # Create supercell to accommodate fractional occupancies
        struct_for_enum = struct.copy()
        effective_sort = sort_by
        
        # Create a supercell to make fractional occupancies work
        try:
            from pymatgen.transformations.standard_transformations import SupercellTransformation
            # Create a supercell based on supercell_size
            scaling_matrix = [[supercell_size, 0, 0], [0, supercell_size, 0], [0, 0, 1]]
            super_trans = SupercellTransformation(scaling_matrix)
            struct_for_enum = super_trans.apply_transformation(struct_for_enum)
        except Exception as e:
            warnings.append(f"Failed to create supercell for '{src_formula}': {e}")
            continue
        
        # Add oxidation states if needed for Ewald ranking  
        needs_oxidation = sort_by == "ewald" and add_oxidation_states
        has_oxidation = False
        
        if needs_oxidation:
            try:
                from pymatgen.analysis.bond_valence import BVAnalyzer
                bva = BVAnalyzer()
                struct_for_enum = bva.get_oxi_state_decorated_structure(struct_for_enum)
                has_oxidation = True
            except Exception as e:
                warnings.append(
                    f"Structure '{src_formula}': could not auto-assign oxidation states "
                    f"({e}). Falling back to sort_by='num_sites' for this structure."
                )
                effective_sort = "num_sites"

        # Use OrderDisorderedStructureTransformation
        # Use no_oxi_states=True if we don't have oxidation states decorated
        trans = OrderDisorderedStructureTransformation(
            algo=0,
            symmetrized_structures=refine_structure,
            no_oxi_states=not has_oxidation,
            symprec=symm_prec if symm_prec else 0.1,
        )

        try:
            raw = trans.apply_transformation(struct_for_enum, return_ranked_list=n_structures)
        except Exception as e:
            warnings.append(f"Ordering failed for '{src_formula}': {e}")
            continue

        # Handle return value - can be Structure or list
        if isinstance(raw, Structure):
            raw = [raw]
        elif not isinstance(raw, list):
            warnings.append(f"Unexpected return type from ordering for '{src_formula}'.")
            continue

        # Calculate Ewald energies if needed for sorting
        structures_with_energy = []
        for s in raw:
            if isinstance(s, dict):
                struct_obj = s.get("structure", s)
            else:
                struct_obj = s
            
            ewald_e = None
            if effective_sort == "ewald":
                try:
                    from pymatgen.analysis.ewald import EwaldSummation
                    ewald = EwaldSummation(struct_obj)
                    ewald_e = ewald.total_energy
                except Exception:
                    pass
            
            structures_with_energy.append({"structure": struct_obj, "energy": ewald_e})

        # Sort structures
        if effective_sort == "ewald":
            structures_with_energy.sort(key=lambda x: x["energy"] if x["energy"] is not None else float('inf'))
        elif effective_sort == "num_sites":
            structures_with_energy.sort(key=lambda x: len(x["structure"]))
        elif sort_by == "random":
            import random as _rng
            _rng.shuffle(structures_with_energy)

        n_atoms_parent = len(struct)
        for entry in structures_with_energy[:n_structures]:
            s = entry["structure"]
            e = entry.get("energy")
            _append_result(
                s, e, src_formula, n_atoms_parent,
                symm_prec, output_format,
                generated_structures, metadata_list, warnings,
                backend="OrderDisorderedStructureTransformation"
            )

    if not generated_structures:
        msg = "No ordered structures were generated."
        if skipped_ordered:
            msg += (
                f" All {len(skipped_ordered)} input structure(s) were fully ordered and skipped. "
                "Set check_ordered_input=False to enumerate ordered structures."
            )
        return {
            "success": False,
            "error": msg,
            "warnings": warnings if warnings else None,
        }

    input_info = {
        "n_input_structures": len(structures),
        "input_formulas": [s.composition.reduced_formula for s in structures],
        "n_skipped_ordered": len(skipped_ordered),
    }

    enumeration_params = {
        "backend": "OrderDisorderedStructureTransformation",
        "supercell_size": supercell_size,
        "n_structures_requested": n_structures,
        "sort_by": sort_by,
        "symm_prec": symm_prec,
        "refine_structure": refine_structure,
        "check_ordered_input": check_ordered_input,
        "add_oxidation_states": add_oxidation_states,
        "output_format": output_format,
    }

    result: Dict[str, Any] = {
        "success": True,
        "count": len(generated_structures),
        "structures": generated_structures,
        "metadata": metadata_list,
        "input_info": input_info,
        "enumeration_params": enumeration_params,
        "message": (
            f"Generated {len(generated_structures)} ordered structure(s) from "
            f"{len(structures)} input structure(s) "
            f"(sort_by='{sort_by}'). Note: No supercell enumeration performed."
        ),
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _append_result(
    ordered_struct,
    ewald_energy,
    src_formula: str,
    n_atoms_parent: int,
    symm_prec: float,
    output_format: str,
    generated_structures: list,
    metadata_list: list,
    warnings: list,
    backend: str,
) -> None:
    """Format an ordered structure and append it (with metadata) to the output lists."""
    sg_number = None
    sg_symbol = None
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(ordered_struct, symprec=symm_prec)
        sg_number = sga.get_space_group_number()
        sg_symbol = sga.get_space_group_symbol()
    except Exception:
        pass

    supercell_size = max(1, round(len(ordered_struct) / n_atoms_parent))

    try:
        if output_format == "poscar":
            from pymatgen.io.vasp import Poscar
            formatted = str(Poscar(ordered_struct))
        elif output_format == "cif":
            from pymatgen.io.cif import CifWriter
            formatted = str(CifWriter(ordered_struct))
        elif output_format == "json":
            from pymatgen.io.cif import CifWriter
            import json
            formatted = json.dumps({"format": "cif", "data": str(CifWriter(ordered_struct))})
        else:
            warnings.append(f"Unknown output_format '{output_format}' — skipping structure.")
            return
    except Exception as e:
        warnings.append(f"Could not format structure (source: '{src_formula}'): {e}. Skipping.")
        return

    meta = {
        "index": len(generated_structures) + 1,
        "source_structure": src_formula,
        "formula": ordered_struct.composition.reduced_formula,
        "n_sites": len(ordered_struct),
        "supercell_size": supercell_size,
        "volume": float(ordered_struct.volume),
        "space_group_number": sg_number,
        "space_group_symbol": sg_symbol,
        "ewald_energy": float(ewald_energy) if ewald_energy is not None else None,
        "is_ordered": ordered_struct.is_ordered,
        "backend": backend,
    }
    generated_structures.append(formatted)
    metadata_list.append(meta)
