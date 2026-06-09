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
                "Creates an isotropic [s, s, s] supercell (all three axes) "
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
    ] = "cif",
    max_atoms: Annotated[
        int,
        Field(
            default=500,
            ge=10,
            le=10000,
            description=(
                "Maximum number of atoms allowed in the supercell after scaling. "
                "The tool searches supercell sizes from 1×1×1 upward to find one where "
                "fractional occupancies can be rounded to integers with error below "
                "composition_tolerance. This caps the search to prevent generating "
                "prohibitively large cells. Default: 500."
            )
        )
    ] = 500,
    composition_tolerance: Annotated[
        float,
        Field(
            default=0.05,
            ge=0.0,
            le=0.5,
            description=(
                "Maximum allowed sum of fractional-atom rounding errors across ALL "
                "disordered sites after supercell scaling (0.0–0.5). "
                "The fallback scales the cell until total_err = "
                "sum(|fractional × n - round(fractional × n)|) ≤ tolerance. "
                "A value of 0.05 means the summed dopant rounding error across all sites "
                "must be ≤0.05 atoms. For low doping on few sites you may need to increase "
                "this to 0.2–0.5. Default: 0.05."
            )
        )
    ] = 0.05,) -> Dict[str, Any]:
    """
    Generate ordered candidates from disordered structures with partial occupancies.

    The workflow is:
    1. Parse one or more disordered input structures.
    2. Expand each structure into an isotropic [s, s, s] supercell.
    3. Use pymatgen's ordering transformation to generate fully ordered candidates.
    4. If enumeration fails (e.g. fractional occupancies don't resolve to integer
       counts), fall back to supercell scaling + rounding (see parameter guidance).
    5. Rank the resulting candidates by Ewald energy, number of sites, or random order.

    The returned structures are fully ordered and suitable for downstream relaxation,
    property prediction, or DFT setup.

    ## Parameter guidance

    **supercell_size** — Creates an isotropic [s, s, s] supercell (all three axes).
    Unlike SQS which scales by atom count, this directly controls the expansion factor.

    **Integer rounding of dopants** — When `OrderDisorderedStructureTransformation`
    fails on the supercell (because fractional occupancies cannot be exactly
    represented as integers), the fallback tries scales 1×–4×, rounding each
    species' count with `round(f × n)`. For low doping on few sites, the rounding
    error may exceed the default `composition_tolerance=0.05`. Example: 3% Mg on
    4 Ga sites has error ~0.16 at scale 3× → a `tolerance_exceeded=True` warning
    is emitted but the structure is returned anyway. Increase `composition_tolerance`
    to 0.2–0.5 to suppress this warning, or pre-scale your input cell before calling.

    **n_structures** — Exhaustive enumeration genuinely produces distinct ordered
    configurations (unlike SQS which may converge seeds to the same minimum).

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
    warnings: List[str] = []
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
    rounding_logs: List[dict] = []

    def _try_enumeration(struct_to_enum, src_formula, sort_override=None):
        """Run OrderDisorderedStructureTransformation on a structure.
        Returns (success_bool, structures_with_energy_list, warnings_list)."""
        enum_sort = sort_override or sort_by
        has_ox = False
        this_struct = struct_to_enum.copy()

        # Add oxidation states if needed for Ewald ranking
        if enum_sort == "ewald" and add_oxidation_states:
            try:
                from pymatgen.analysis.bond_valence import BVAnalyzer
                bva = BVAnalyzer()
                this_struct = bva.get_oxi_state_decorated_structure(this_struct)
                has_ox = True
            except Exception as e:
                warnings.append(
                    f"'{src_formula}': could not assign oxidation states ({str(e)[:60]}). "
                    f"Falling back to sort_by='num_sites'."
                )
                enum_sort = "num_sites"

        trans = OrderDisorderedStructureTransformation(
            algo=0,
            symmetrized_structures=refine_structure,
            no_oxi_states=not has_ox,
            symprec=symm_prec if symm_prec else 0.1,
        )
        try:
            raw = trans.apply_transformation(this_struct, return_ranked_list=n_structures)
        except Exception as e:
            return False, [], [f"Ordering failed for '{src_formula}': {e}"]

        if isinstance(raw, Structure):
            raw = [raw]
        elif not isinstance(raw, list):
            return False, [], [f"Unexpected return type for '{src_formula}'."]

        results = []
        for s in raw:
            s_obj = s.get("structure", s) if isinstance(s, dict) else s
            ew = None
            if enum_sort == "ewald":
                try:
                    from pymatgen.analysis.ewald import EwaldSummation
                    ew = EwaldSummation(s_obj).total_energy
                except Exception:
                    pass
            results.append({"structure": s_obj, "energy": ew})

        if enum_sort == "ewald":
            results.sort(key=lambda x: x["energy"] if x["energy"] is not None else float('inf'))
        elif enum_sort == "num_sites":
            results.sort(key=lambda x: len(x["structure"]))
        else:
            import random as _rng
            _rng.shuffle(results)

        return True, results, []

    def _supercell_round_fallback(struct):
        """When enumeration fails on a disordered structure, find a supercell
        size where rounding fractional occupancies → integers makes enumeration
        succeed. Tries scales 1-4, uses the first acceptable one or the best
        available. Returns (scaled_rounded_structure, scale, was_rounded, log)."""
        from copy import deepcopy
        from collections import defaultdict

        log = {"applied": False, "scale": 1, "error": 0.0}
        if struct.is_ordered:
            return struct, 1, False, log

        # Build site-group signatures
        sig_groups = defaultdict(list)
        for i, site in enumerate(struct):
            if not site.is_ordered:
                sig = tuple(sorted((str(sp), round(float(f), 6)) for sp, f in site.species.items()))
                sig_groups[sig].append(i)

        best_result = None
        best_scale = 1
        best_err = float("inf")

        for scale in [1, 2, 3, 4]:
            if len(struct) * (scale ** 3) > max_atoms:
                continue
            total_err = 0.0
            for sig, idxs in sig_groups.items():
                if len(sig) <= 1: continue
                occ = {sp: f for sp, f in sig}
                n = len(idxs) * (scale ** 3)
                for sp, f in occ.items():
                    total_err += abs(f * n - round(f * n))
            
            # Build the rounded structure for this scale
            scaled = deepcopy(struct)
            scaled.make_supercell([scale, scale, scale])
            sc_groups = defaultdict(list)
            for idx, site in enumerate(scaled):
                if not site.is_ordered:
                    sig = tuple(sorted((str(sp), round(float(f), 6)) for sp, f in site.species.items()))
                    sc_groups[sig].append(idx)
            for sig, idxs in sc_groups.items():
                if len(sig) <= 1: continue
                occ = {sp: f for sp, f in sig}
                n = len(idxs)
                counts = {sp: round(f * n) for sp, f in occ.items()}
                diff = n - sum(counts.values())
                if diff != 0:
                    adj = sorted(occ.items(), key=lambda x: abs(x[1] * n - round(x[1] * n)), reverse=True)
                    for k in range(abs(diff)):
                        counts[adj[k % len(adj)][0]] += 1 if diff > 0 else -1
                assn = []
                for sp, cnt in counts.items():
                    assn.extend([sp] * cnt)
                for i2, sp in zip(idxs, assn):
                    scaled.replace(i2, sp)

            if total_err <= composition_tolerance:
                # Acceptable — return immediately
                log = {"applied": True, "scale": scale, "error": round(total_err, 4)}
                return scaled, scale, True, log
            
            if total_err < best_err:
                best_result = scaled
                best_scale = scale
                best_err = total_err

        if best_result is not None:
            # Return best available even if tolerance exceeded
            log = {"applied": True, "scale": best_scale, "error": round(best_err, 4),
                   "tolerance_exceeded": True}
            warnings.append(
                f"Rounding error {best_err:.4f} exceeds composition_tolerance "
                f"({composition_tolerance}) at best scale {best_scale}×."
            )
            return best_result, best_scale, True, log

        log["reason"] = "No scale within max_atoms"
        return struct, 1, False, log

    for i, struct in enumerate(structures):
        src_formula = struct.composition.reduced_formula
        effective_sort = sort_by

        # Skip already-ordered structures
        if struct.is_ordered:
            if check_ordered_input:
                skipped_ordered.append(src_formula)
                warnings.append(
                    f"'{src_formula}' is already fully ordered and was skipped. "
                    "Set check_ordered_input=False to enumerate it anyway."
                )
                continue
            # If ordered and check_ordered_input=False, pass through
            struct_for_enum = struct.copy()
        else:
            struct_for_enum = struct.copy()
            # Create supercell
            try:
                from pymatgen.transformations.standard_transformations import SupercellTransformation
                sm = [[supercell_size, 0, 0], [0, supercell_size, 0], [0, 0, supercell_size]]
                struct_for_enum = SupercellTransformation(sm).apply_transformation(struct_for_enum)
            except Exception as e:
                warnings.append(f"Supercell failed for '{src_formula}': {e}")
                continue

        # Try enumeration
        enum_ok, results, enum_warns = _try_enumeration(struct_for_enum, src_formula)
        if enum_warns:
            warnings.extend(enum_warns)

        if not enum_ok and not struct.is_ordered:
            # Enumeration failed — try supercell scaling + rounding fallback
            fb_struct, fb_scale, was_rounded, fb_log = _supercell_round_fallback(struct)
            if was_rounded:
                rounding_logs.append(fb_log)
                warnings.append(
                    f"'{src_formula}': enumeration failed on original structure. "
                    f"Applied supercell scaling {fb_scale}× + rounding (error={fb_log['error']:.4f})."
                )
                # Re-try with the scaled+rounded (now fully ordered) structure
                _append_result(
                    fb_struct, None, src_formula, len(struct),
                    symm_prec, output_format,
                    generated_structures, metadata_list, warnings,
                    backend="enumeration_orderer (via rounding fallback)"
                )
                continue
            else:
                # Fallback also failed — just record the original error
                if not enum_warns:
                    warnings.append(f"All ordering attempts failed for '{src_formula}'.")
                continue

        n_atoms_parent = len(struct)
        for entry in results[:n_structures]:
            s_obj = entry["structure"]
            e = entry.get("energy")
            _append_result(
                s_obj, e, src_formula, n_atoms_parent,
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
        "max_atoms": max_atoms,
        "composition_tolerance": composition_tolerance,
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
            f"(sort_by='{sort_by}')."
        ),
    }
    if warnings:
        result["warnings"] = warnings
    if any(log.get("applied") for log in rounding_logs):
        result["rounding_preprocessing"] = rounding_logs
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
