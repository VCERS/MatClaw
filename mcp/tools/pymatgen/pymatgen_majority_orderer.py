"""
Approximate a disordered structure by keeping the majority species on each mixed site.

This tool converts structures with partial occupancies into a single fully ordered
structure by replacing each disordered site with its dominant species. It preserves
the original cell and site count: no supercell expansion, no configuration search,
and no statistical optimisation.

Best for:
    - Dilute doping where minority species act as a small perturbation to the host
    - Fast screening workflows that require compact ordered cells
    - Converting disorder_generator output into compact ordered structures for downstream screening

Not appropriate for:
    - Site-specific dopant studies where the arrangement of minority species matters
    - Concentrated solid solutions where dopant-dopant correlations are important
    - Cases where disorder is itself the property of interest

Relationship to the other orderers:
    - pymatgen_majority_orderer: one compact approximation, no supercell
    - pymatgen_enumeration_orderer: many ordered candidates within a chosen supercell
    - pymatgen_sqs_orderer: one or more quasirandom ordered supercells for random alloys
"""

from typing import Dict, Any, Optional, List, Union, Annotated
from pydantic import Field


def pymatgen_majority_orderer(
    input_structures: Annotated[
        Union[str, List[str]],
        Field(
            description=(
                "Input structure(s) with fractional site occupancies (disordered). "
                "Accepts CIF string or list of CIF strings. "
                "Each structure may be disordered or already fully ordered. Ordered "
                "inputs are returned unchanged."
            )
        )
    ],
    check_ordered_input: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Controls whether already ordered structures emit a warning. "
                "If True (default), ordered inputs are returned unchanged and a warning "
                "is added. If False, they are returned unchanged without that warning."
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
                "'json': JSON-serialised Structure dict string. "
                "Default: 'cif'."
            )
        )
    ] = "cif"
) -> Dict[str, Any]:
    """
    Convert disordered structures to a single ordered approximation.

    This tool applies the majority-species approximation independently on each mixed
    site: the species with the highest occupancy is kept and all minority species on
    that site are removed. The lattice, number of sites, and overall cell size are
    preserved.

    This makes the tool useful for dilute-doping screening, but it also means the
    output should be treated as an approximation when the removed minority species are
    expected to affect local structure or energetics.

    Example:
        Input:  Sr0.99Sm0.01Nb2O6
        Output: SrNb2O6
        Interpretation: minority Sm is dropped to produce a compact screening cell.

    Returns:
        dict:
            success             (bool)  Whether processing succeeded.
            count               (int)   Number of returned structures (same as input count).
            structures          (list)  Ordered structures in the requested output_format.
            metadata            (list)  Per-structure metadata:
                index               (int)   Sequential index (1-based).
                source_formula      (str)   Reduced formula of the input structure.
                ordered_formula     (str)   Reduced formula after majority-species approximation.
                n_sites             (int)   Number of sites in the returned structure.
                volume              (float) Cell volume in Å³.
                space_group_number  (int)   Space group number (if determinable).
                space_group_symbol  (str)   Hermann-Mauguin symbol (if determinable).
                was_disordered      (bool)  Whether the input had partial occupancies.
                sites_converted     (int)   Number of mixed sites simplified.
                lost_species        (list)  Minority species removed during simplification.
            input_info          (dict)  Summary of the input structures.
            ordering_params     (dict)  Parameters used for this run.
            message             (str)   Human-readable summary.
            warnings            (list)  Non-fatal warnings (if any).
            error               (str)   Error message if success=False.
    """
    try:
        from pymatgen.core import Structure, PeriodicSite
        from pymatgen.io.cif import CifWriter
        from pymatgen.io.vasp import Poscar
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
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
                    "error": f"Input structure {i} must be a CIF string, got {type(item).__name__}."
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to parse input structure {i}: {e}"}
    
    if not structures:
        return {"success": False, "error": "No valid input structures provided."}
    
    # Process structures
    ordered_structures: List[Structure] = []
    metadata_list: List[Dict[str, Any]] = []
    warnings: List[str] = []
    
    for idx, struct in enumerate(structures):
        src_formula = struct.composition.reduced_formula
        
        # Check if already ordered
        if struct.is_ordered:
            # Add warning if check_ordered_input is True
            if check_ordered_input:
                warnings.append(
                    f"Structure '{src_formula}' is already fully ordered (no partial occupancies). "
                    "Passing through unchanged. Set check_ordered_input=False to suppress this warning."
                )
            
            ordered_structures.append(struct)
            
            # Create metadata for already-ordered structure
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                sg_num = sga.get_space_group_number()
                sg_symbol = sga.get_space_group_symbol()
            except:
                sg_num = None
                sg_symbol = "unknown"
            
            metadata_list.append({
                "index": len(metadata_list) + 1,
                "source_formula": src_formula,
                "ordered_formula": src_formula,
                "n_sites": len(struct),
                "volume": round(struct.volume, 4),
                "space_group_number": sg_num,
                "space_group_symbol": sg_symbol,
                "was_disordered": False,
                "sites_converted": 0,
                "lost_species": []
            })
            continue
        
        # Apply majority-species approximation
        ordered_sites = []
        sites_converted = 0
        all_lost_species = set()
        
        for site in struct:
            if len(site.species) > 1:
                # Disordered site - take dominant species
                sites_converted += 1
                dominant_species, dominant_occ = max(site.species.items(), key=lambda x: x[1])
                
                # Track lost species
                for species, occ in site.species.items():
                    if species != dominant_species:
                        all_lost_species.add(str(species))
                
                # Create new ordered site
                ordered_site = PeriodicSite(
                    dominant_species,
                    site.frac_coords,
                    site.lattice,
                    properties=site.properties
                )
                ordered_sites.append(ordered_site)
            else:
                # Already ordered site - keep as is
                ordered_sites.append(site)
        
        # Create ordered structure
        ordered_struct = Structure.from_sites(ordered_sites)
        ordered_structures.append(ordered_struct)
        
        # Get space group info
        try:
            sga = SpacegroupAnalyzer(ordered_struct, symprec=0.1)
            sg_num = sga.get_space_group_number()
            sg_symbol = sga.get_space_group_symbol()
        except:
            sg_num = None
            sg_symbol = "unknown"
        
        # Store metadata
        metadata_list.append({
            "index": len(metadata_list) + 1,
            "source_formula": src_formula,
            "ordered_formula": ordered_struct.composition.reduced_formula,
            "n_sites": len(ordered_struct),
            "volume": round(ordered_struct.volume, 4),
            "space_group_number": sg_num,
            "space_group_symbol": sg_symbol,
            "was_disordered": True,
            "sites_converted": sites_converted,
            "lost_species": sorted(list(all_lost_species))
        })
        
        # Add info warning if significant species lost
        if all_lost_species:
            total_sites = len(struct)
            lost_str = ", ".join(sorted(all_lost_species))
            warnings.append(
                f"Structure '{src_formula}': Removed minority species [{lost_str}] "
                f"from {sites_converted}/{total_sites} sites. "
                f"Valid for dilute doping; for >10% concentrations, consider enumeration_orderer or sqs_orderer."
            )
    
    # Convert to requested output format
    output_structures = []
    for struct in ordered_structures:
        if output_format == "cif":
            writer = CifWriter(struct)
            output_structures.append(str(writer))
        elif output_format == "poscar":
            poscar = Poscar(struct)
            output_structures.append(poscar.get_string())
        elif output_format == "json":
            output_structures.append(struct.as_dict())
    
    # Build response
    result = {
        "success": True,
        "count": len(output_structures),
        "structures": output_structures,
        "metadata": metadata_list,
        "input_info": {
            "n_inputs": len(structures),
            "input_formulas": [s.composition.reduced_formula for s in structures]
        },
        "ordering_params": {
            "method": "majority_species_approximation",
            "supercell_expansion": False,
            "check_ordered_input": check_ordered_input,
            "output_format": output_format
        },
        "message": f"Successfully converted {len(output_structures)} structure(s) using majority-species approximation."
    }
    
    if warnings:
        result["warnings"] = warnings
    
    return result
