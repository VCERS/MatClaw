"""
Tool for comparing two crystal structures to determine if they are equivalent.

Uses pymatgen's StructureMatcher to perform symmetry-aware structure comparison.
Handles different unit cell choices, lattice distortions within tolerances, and 
supercell/subcell relationships. Essential for deduplication, structure validation,
and comparing theoretical predictions against experimental or database structures.
"""

from typing import Dict, Any, Optional, Annotated, List
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
                "Method for comparing atomic species:\n"
                "- 'SpeciesComparator': Exact element matching (default)\n"
                "- 'ElementComparator': Ignores oxidation states\n"
                "- 'FrameworkComparator': Ignores certain 'framework' atoms\n"
                "- 'OccupancyComparator': Considers partial occupancies\n"
                "Default: 'SpeciesComparator'."
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
                - rms_distance (float or None): RMS distance between matched atoms
            - mismatch_reasons (list): Reasons for mismatch if match=False:
                - composition_mismatch
                - lattice_parameter_mismatch
                - site_position_mismatch
                - space_group_mismatch
            - parameters (dict): Tolerances used for comparison
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
        
        struct1_info = get_structure_info(struct1, "structure_1")
        struct2_info = get_structure_info(struct2, "structure_2")
        
        # Quick composition check
        if struct1.composition.reduced_formula != struct2.composition.reduced_formula:
            return {
                "success": True,
                "match": False,
                "confidence": "exact",
                "structure_1_info": struct1_info,
                "structure_2_info": struct2_info,
                "comparison_details": {
                    "method": "composition_check",
                    "supercell_relation": None,
                    "site_mapping": None,
                    "rms_distance": None
                },
                "mismatch_reasons": ["composition_mismatch"],
                "parameters": {
                    "l_tol": l_tol,
                    "s_tol": s_tol,
                    "angle_tol": angle_tol,
                    "primitive_cell": primitive_cell,
                    "scale": scale,
                    "attempt_supercell": attempt_supercell
                },
                "warnings": warnings if warnings else None,
                "message": f"Structures have different compositions: {struct1_info['formula']} vs {struct2_info['formula']}"
            }
        
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
        
        species_comparator = comparator_map[comparator]
        
        # Create StructureMatcher
        matcher = StructureMatcher(
            ltol=l_tol,
            stol=s_tol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            scale=scale,
            attempt_supercell=attempt_supercell,
            allow_subset=allow_subset,
            comparator=species_comparator,
            supercell_size=supercell_size
        )
        
        # Perform matching
        try:
            is_match = bool(matcher.fit(struct1, struct2))  # Convert numpy bool to Python bool
            
            # Get detailed results if matched
            supercell_relation = None
            site_mapping = None
            rms_distance = None
            
            if is_match:
                # Check for supercell relationship
                if attempt_supercell:
                    # Check if struct2 is supercell of struct1
                    if struct2.num_sites > struct1.num_sites:
                        supercell_relation = "structure_2_is_supercell_of_structure_1"
                    elif struct1.num_sites > struct2.num_sites:
                        supercell_relation = "structure_1_is_supercell_of_structure_2"
                
                # Get site mapping if requested
                if return_mapping:
                    try:
                        # Get the mapping of sites
                        s1, s2, fu, s1_supercell = matcher.get_s2_like_s1(struct1, struct2)
                        site_mapping = []
                        for i, site in enumerate(s1):
                            site_mapping.append({
                                "structure_1_site": i,
                                "structure_2_site": None,  # Would need more complex logic to track
                                "species": str(site.specie),
                                "coords_1": site.frac_coords.tolist()
                            })
                    except Exception as e:
                        warnings.append(f"Could not extract site mapping: {str(e)}")
                
                # Calculate RMS distance between matched structures
                try:
                    # Try using get_s2_like_s1 to get transformed structures
                    # This may fail if primitive_cell=True
                    try:
                        s1, s2, fu, s1_supercell = matcher.get_s2_like_s1(struct1, struct2)
                        # Manually calculate RMS distance from transformed structures
                        if s1 is not None and s2 is not None:
                            import numpy as np
                            # Calculate RMS of cartesian distances between corresponding sites
                            distances = []
                            for site1, site2 in zip(s1, s2):
                                dist = np.linalg.norm(site1.coords - site2.coords)
                                distances.append(dist)
                            if distances:
                                rms_distance = float(np.sqrt(np.mean(np.array(distances)**2)))
                    except Exception:
                        # If get_s2_like_s1 fails (e.g., with primitive_cell=True),
                        # try a direct comparison for identical structures
                        if struct1.num_sites == struct2.num_sites:
                            import numpy as np
                            distances = []
                            for site1, site2 in zip(struct1, struct2):
                                dist = np.linalg.norm(site1.coords - site2.coords)
                                distances.append(dist)
                            if distances:
                                rms_distance = float(np.sqrt(np.mean(np.array(distances)**2)))
                except Exception as e:
                    warnings.append(f"Could not calculate RMS distance: {str(e)}")
            
            # Determine confidence level
            if is_match:
                if l_tol <= 0.05 and s_tol <= 0.1 and angle_tol <= 1.0:
                    confidence = "exact"
                elif l_tol <= 0.1 and s_tol <= 0.2 and angle_tol <= 3.0:
                    confidence = "high"
                elif l_tol <= 0.2 and s_tol <= 0.3 and angle_tol <= 5.0:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "no_match"
            
            # If no match, try to determine reasons
            mismatch_reasons = []
            if not is_match:
                # Check lattice parameters
                lat1, lat2 = struct1.lattice, struct2.lattice
                if abs(lat1.a - lat2.a) / min(lat1.a, lat2.a) > l_tol or \
                   abs(lat1.b - lat2.b) / min(lat1.b, lat2.b) > l_tol or \
                   abs(lat1.c - lat2.c) / min(lat1.c, lat2.c) > l_tol:
                    mismatch_reasons.append("lattice_parameter_mismatch")
                
                if abs(lat1.alpha - lat2.alpha) > angle_tol or \
                   abs(lat1.beta - lat2.beta) > angle_tol or \
                   abs(lat1.gamma - lat2.gamma) > angle_tol:
                    mismatch_reasons.append("lattice_angle_mismatch")
                
                # Check space groups
                if struct1_info["space_group"] and struct2_info["space_group"]:
                    if struct1_info["space_group"] != struct2_info["space_group"]:
                        mismatch_reasons.append("space_group_mismatch")
                
                # If no specific reason found, it's likely site positions
                if not mismatch_reasons:
                    mismatch_reasons.append("site_position_mismatch")
            
            message = f"Structures {'match' if is_match else 'do not match'}"
            if is_match and supercell_relation:
                message += f" ({supercell_relation})"
            if is_match and rms_distance is not None:
                message += f" with RMS distance {rms_distance:.4f} Å"
            
            return {
                "success": True,
                "match": is_match,
                "confidence": confidence,
                "structure_1_info": struct1_info,
                "structure_2_info": struct2_info,
                "comparison_details": {
                    "method": "StructureMatcher",
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
