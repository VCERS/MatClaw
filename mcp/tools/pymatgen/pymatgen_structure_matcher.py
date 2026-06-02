"""
Tool for comparing two crystal structures to determine if they are equivalent.

Uses pymatgen's StructureMatcher to perform symmetry-aware structure comparison.
Handles different unit cell choices, lattice distortions within tolerances, and
supercell/subcell relationships. Essential for deduplication, structure validation,
and comparing theoretical predictions against experimental or database structures.

The 'comparator' parameter controls the matching criteria, allowing flexible
comparison strategies from strict composition matching to geometry-only comparisons.
This enables comparing ordered structures to disordered ones, structures with
partial occupancy differences, or purely geometric frameworks.
"""

from typing import Dict, Any, Annotated
from pydantic import Field


def pymatgen_structure_matcher(
    structure_1: Annotated[
        str,
        Field(
            description=(
                "First crystal structure to compare. Accepted formats:\n"
                "- CIF string\n"
                "- POSCAR/CONTCAR string\n"
                "Output from pymatgen tools or Materials Project API can be passed directly."
            )
        )
    ],
    structure_2: Annotated[
        str,
        Field(
            description=(
                "Second crystal structure to compare. Accepted formats:\n"
                "- CIF string\n"
                "- POSCAR/CONTCAR string\n"
                "Must be same format type as structure_1 or compatible."
            )
        )
    ],
    l_tol: Annotated[
        float,
        Field(
            default=0.2,
            ge=0.0,
            le=1.0,
            description=(
                "Fractional length tolerance for lattice parameter matching.\n"
                "Structures match if |a1-a2|/min(a1,a2) < l_tol for all lattice vectors.\n"
                "Typical values: 0.2 (default, permissive), 0.1 (moderate), 0.05 (strict)."
            )
        )
    ] = 0.2,
    s_tol: Annotated[
        float,
        Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description=(
                "Site tolerance for atomic position matching (Angstroms).\n"
                "Maximum distance atoms can be apart to be considered equivalent.\n"
                "Typical values: 0.3 (default), 0.2 (moderate), 0.1 (strict)."
            )
        )
    ] = 0.3,
    angle_tol: Annotated[
        float,
        Field(
            default=5.0,
            ge=0.0,
            le=45.0,
            description=(
                "Angle tolerance for lattice angle matching (degrees).\n"
                "Structures match if |α1-α2| < angle_tol for all angles.\n"
                "Typical values: 5.0 (default), 3.0 (moderate), 1.0 (strict)."
            )
        )
    ] = 5.0,
    primitive_cell: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True, reduces structures to primitive cells before comparison.\n"
                "Recommended for most cases to avoid issues with different cell choices.\n"
                "Set to False if you want to preserve conventional cell differences."
            )
        )
    ] = True,
    scale: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True, scales volumes to be equal before comparison.\n"
                "Useful when comparing structures at different pressures/temperatures.\n"
                "Set to False for strict volume matching."
            )
        )
    ] = True,
    attempt_supercell: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, checks if one structure is a supercell of the other.\n"
                "More computationally expensive but catches supercell relationships.\n"
                "Useful when comparing experimental (supercell) vs primitive cells."
            )
        )
    ] = False,
    allow_subset: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, allows matching if one structure is a subset of the other.\n"
                "Useful for comparing structures with vacancies or partial occupancy."
            )
        )
    ] = False,
    comparator: Annotated[
        str,
        Field(
            default="SpeciesComparator",
            description=(
                "Method for comparing atomic species. Controls how strictly the matching criteria are applied:\n"
                "\n"
                "• 'SpeciesComparator' (default):\n"
                "    Exact element and oxidation state matching.\n"
                "    Fails if compositions differ (even slightly).\n"
                "    Use for: Strict structural matching where composition must match exactly.\n"
                "    Example: Distinguishing Fe2+ from Fe3+.\n"
                "\n"
                "• 'ElementComparator':\n"
                "    Elements must match, but ignores oxidation states.\n"
                "    Still fails if element compositions differ.\n"
                "    Use for: Comparing same elements with different oxidation states.\n"
                "    Example: Fe2O3 vs FeO both match (same elements, Fe and O).\n"
                "\n"
                "• 'FrameworkComparator':\n"
                "    Ignores ALL chemistry - compares geometry only!\n"
                "    Succeeds even with completely different compositions.\n"
                "    Use for: Comparing ordered vs disordered structures, different substitutions, or purely geometric frameworks.\n"
                "    Example: Ba₀.₉₄₅Mg₀.₁Ga₃.₉₅₅Se₇ (disordered) matches BaMgGa₄Se₇ (ordered) geometrically.\n"
                "\n"
                "• 'OccupancyComparator':\n"
                "    Matches occupancy PATTERNS on sites (full vs partial).\n"
                "    Fails when occupancy patterns differ (e.g., fractional vs integer).\n"
                "    Use for: Comparing structures with similar partial occupancy distributions.\n"
                "    Example: Matches structures with 0.5 occupancy at same sites, but fails if one has 0.5 and other has 1.0.\n"
            )
        )
    ] = "SpeciesComparator",
    supercell_size: Annotated[
        str,
        Field(
            default="num_sites",
            description=(
                "How to determine supercell size for attempt_supercell:\n"
                "- 'num_sites': Based on number of sites (faster, default)\n"
                "- 'volume': Based on volume ratio (more thorough)\n"
                "Only used when attempt_supercell=True."
            )
        )
    ] = "num_sites",
    return_mapping: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True and structures match, returns the site-to-site mapping.\n"
                "Useful for tracking atom correspondences between structures."
            )
        )
    ] = False
) -> Dict[str, Any]:
    """
    Compare two crystal structures to determine if they are equivalent.
    
    Uses symmetry-aware comparison that accounts for:
    - Different unit cell choices (primitive vs conventional)
    - Small lattice distortions within tolerances
    - Different atomic orderings or origin shifts
    - Optionally: supercell/subcell relationships
    
    Perfect for:
    - Deduplicating structure databases
    - Validating structure prediction results
    - Comparing DFT relaxed structures to experimental ones
    - Checking if two CIF files represent the same material
    
    Returns:
        dict: Comparison results containing:
            - success (bool): Whether comparison completed successfully
            - match (bool): Whether structures are equivalent within tolerances
            - confidence (str): "exact", "high", "medium", or "low" based on tolerances
            - rms_distance (float or None): RMS displacement normalized by (Vol/nsites)^(1/3) (Å)
            - max_distance (float or None): Maximum distance between paired sites (Å)
            - structure_1_info (dict): Information about first structure:
                - formula (str): Reduced chemical formula
                - n_sites (int): Number of sites
                - space_group (int): Space group number
                - lattice (dict): Lattice parameters {a, b, c, alpha, beta, gamma, volume}
                - is_ordered (bool): Whether structure has full occupancy
            - structure_2_info (dict): Information about second structure
            - comparison_details (dict): Detailed comparison information:
                - method (str): Comparison method used
                - supercell_relation (str or None): If one is supercell of other
                - site_mapping (list or None): Site correspondences (if return_mapping=True)
                - rms_distance (float or None): RMS distance between structures (Å)
            - mismatch_reasons (list): Reasons for mismatch if match=False:
                - composition_mismatch
                - lattice_parameter_mismatch
                - site_position_mismatch
                - space_group_mismatch
            - parameters (dict): Tolerances and comparator used for comparison
            - warnings (list): Any warnings generated
            - error (str): Error message if comparison failed
    """
    
    try:
        try:
            from pymatgen.core import Structure
            from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator, FrameworkComparator, OccupancyComparator, SpeciesComparator
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            import numpy as np
        except ImportError as e:
            return {
                "success": False,
                "error": f"Required library not installed: {str(e)}. Install with: pip install pymatgen"
            }
        
        warnings = []
        
        # Parse structures
        def parse_structure(structure_str: str, label: str) -> Structure:
            """Helper to parse structure from string."""
            structure_str = structure_str.strip()
            
            # Try CIF first
            if "data_" in structure_str or "_cell_length_a" in structure_str:
                try:
                    from pymatgen.io.cif import CifParser
                    from io import StringIO
                    parser = CifParser(StringIO(structure_str))
                    return parser.get_structures()[0]
                except Exception as e:
                    raise ValueError(f"Failed to parse {label} as CIF: {str(e)}")
            
            # Try POSCAR
            else:
                try:
                    from pymatgen.io.vasp import Poscar
                    from io import StringIO
                    poscar = Poscar.from_str(structure_str)
                    return poscar.structure
                except Exception as e:
                    raise ValueError(f"Failed to parse {label} as POSCAR: {str(e)}")
        
        try:
            struct1 = parse_structure(structure_1, "structure_1")
            struct2 = parse_structure(structure_2, "structure_2")
        except Exception as e:
            return {
                "success": False,
                "error": f"Structure parsing failed: {str(e)}"
            }
        
        def determine_confidence(is_match: bool) -> str:
            if not is_match:
                return "no_match"
            if l_tol <= 0.05 and s_tol <= 0.1 and angle_tol <= 1.0:
                return "exact"
            if l_tol <= 0.1 and s_tol <= 0.2 and angle_tol <= 3.0:
                return "high"
            if l_tol <= 0.2 and s_tol <= 0.3 and angle_tol <= 5.0:
                return "medium"
            return "low"

        # Get structure information
        def get_structure_info(struct: Structure, label: str) -> Dict[str, Any]:
            """Extract comprehensive structure information."""
            info = {
                "formula": struct.composition.reduced_formula,
                "n_sites": struct.num_sites,
                "lattice": {
                    "a": struct.lattice.a,
                    "b": struct.lattice.b,
                    "c": struct.lattice.c,
                    "alpha": struct.lattice.alpha,
                    "beta": struct.lattice.beta,
                    "gamma": struct.lattice.gamma,
                    "volume": struct.lattice.volume
                },
                "is_ordered": struct.is_ordered
            }
            
            # Try to get space group
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                info["space_group"] = sga.get_space_group_number()
                info["space_group_symbol"] = sga.get_space_group_symbol()
            except Exception as e:
                warnings.append(f"Could not determine space group for {label}: {str(e)}")
                info["space_group"] = None
                info["space_group_symbol"] = None
            
            return info

        def build_framework_structure(struct: Structure) -> Structure:
            """Return a chemistry-agnostic structure preserving only the site framework."""
            return Structure(
                lattice=struct.lattice,
                species=["H"] * struct.num_sites,
                coords=[site.frac_coords for site in struct],
                coords_are_cartesian=False,
            )

        def get_supercell_relation(struct_a: Structure, struct_b: Structure) -> str | None:
            if not attempt_supercell:
                return None
            if struct_b.num_sites > struct_a.num_sites:
                return "structure_2_is_supercell_of_structure_1"
            if struct_a.num_sites > struct_b.num_sites:
                return "structure_1_is_supercell_of_structure_2"
            return None

        def infer_mismatch_reasons(
            struct_a: Structure,
            struct_b: Structure,
            info_a: Dict[str, Any],
            info_b: Dict[str, Any],
            *,
            include_composition: bool,
        ) -> list[str]:
            mismatch_reasons = []
            if include_composition and struct_a.composition.reduced_formula != struct_b.composition.reduced_formula:
                mismatch_reasons.append("composition_mismatch")

            lat1, lat2 = struct_a.lattice, struct_b.lattice
            if abs(lat1.a - lat2.a) / min(lat1.a, lat2.a) > l_tol or \
               abs(lat1.b - lat2.b) / min(lat1.b, lat2.b) > l_tol or \
               abs(lat1.c - lat2.c) / min(lat1.c, lat2.c) > l_tol:
                mismatch_reasons.append("lattice_parameter_mismatch")

            if abs(lat1.alpha - lat2.alpha) > angle_tol or \
               abs(lat1.beta - lat2.beta) > angle_tol or \
               abs(lat1.gamma - lat2.gamma) > angle_tol:
                mismatch_reasons.append("lattice_angle_mismatch")

            if info_a["space_group"] and info_b["space_group"]:
                if info_a["space_group"] != info_b["space_group"]:
                    mismatch_reasons.append("space_group_mismatch")

            if not mismatch_reasons:
                mismatch_reasons.append("site_position_mismatch")

            return mismatch_reasons
        
        struct1_info = get_structure_info(struct1, "structure_1")
        struct2_info = get_structure_info(struct2, "structure_2")

        
        # Set up comparator
        comparator_map = {
            "SpeciesComparator": SpeciesComparator(),
            "ElementComparator": ElementComparator(),
            "FrameworkComparator": FrameworkComparator(),
            "OccupancyComparator": OccupancyComparator()
        }
        
        if comparator not in comparator_map:
            return {
                "success": False,
                "error": f"Unknown comparator: {comparator}. Choose from: {list(comparator_map.keys())}"
            }

        exact_comparator = comparator_map[comparator]

        # Create StructureMatchers
        matcher = StructureMatcher(
            ltol=l_tol,
            stol=s_tol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            scale=scale,
            attempt_supercell=attempt_supercell,
            allow_subset=allow_subset,
            comparator=exact_comparator,
            supercell_size=supercell_size
        )
        
        # Perform matching
        try:
            # With FrameworkComparator, skip composition check since it's geometry-only
            if comparator == "FrameworkComparator":
                is_match = bool(matcher.fit(struct1, struct2))
                comparison_method = "StructureMatcher"
                mismatch_reasons = None
            else:
                compositions_match = struct1.composition.reduced_formula == struct2.composition.reduced_formula
                if compositions_match:
                    is_match = bool(matcher.fit(struct1, struct2))
                    comparison_method = "StructureMatcher"
                    mismatch_reasons = None
                else:
                    is_match = False
                    comparison_method = "composition_check"
                    mismatch_reasons = ["composition_mismatch"]

            supercell_relation = get_supercell_relation(struct1, struct2) if is_match else None
            site_mapping = None
            # Calculate RMS distance for diagnostic purposes
            try:
                rms_result_raw = matcher.get_rms_dist(struct1, struct2)
                if rms_result_raw:
                    normalized_rms, max_distance = rms_result_raw
                    rms_distance = float(normalized_rms)
                    max_distance = float(max_distance)
                else:
                    rms_distance = None
                    max_distance = None
            except Exception as e:
                warnings.append(f"Could not calculate RMS distance: {str(e)}")
                rms_distance = None
                max_distance = None

            if is_match and return_mapping:
                try:
                    s1, s2, fu, s1_supercell = matcher.get_s2_like_s1(struct1, struct2)
                    site_mapping = []
                    for i, site in enumerate(s1):
                        site_mapping.append({
                            "structure_1_site": i,
                            "structure_2_site": None,
                            "species": str(site.specie),
                            "coords_1": site.frac_coords.tolist()
                        })
                except Exception as e:
                    warnings.append(f"Could not extract site mapping: {str(e)}")

            confidence = determine_confidence(is_match)
            if not is_match and mismatch_reasons is None:
                mismatch_reasons = infer_mismatch_reasons(
                    struct1,
                    struct2,
                    struct1_info,
                    struct2_info,
                    include_composition=True,
                )
            
            message = f"Structures {'match' if is_match else 'do not match'}"
            if is_match and supercell_relation:
                message += f" ({supercell_relation})"
            if is_match and rms_distance is not None:
                message += f" with RMS distance {rms_distance:.4f} Å"
            
            return {
                "success": True,
                "match": is_match,
                "confidence": confidence,
                "rms_distance": rms_distance,
                "max_distance": max_distance,
                "structure_1_info": struct1_info,
                "structure_2_info": struct2_info,
                "comparison_details": {
                    "method": comparison_method,
                    "supercell_relation": supercell_relation,
                    "site_mapping": site_mapping,
                    "rms_distance": rms_distance
                },
                "mismatch_reasons": mismatch_reasons if not is_match else None,
                "parameters": {
                    "l_tol": l_tol,
                    "s_tol": s_tol,
                    "angle_tol": angle_tol,
                    "primitive_cell": primitive_cell,
                    "scale": scale,
                    "attempt_supercell": attempt_supercell,
                    "allow_subset": allow_subset,
                    "comparator": comparator,
                    "supercell_size": supercell_size
                },
                "warnings": warnings if warnings else None,
                "message": message
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Structure matching failed: {str(e)}",
                "error_type": type(e).__name__,
                "structure_1_info": struct1_info,
                "structure_2_info": struct2_info
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error in structure matcher: {str(e)}",
            "error_type": type(e).__name__
        }
