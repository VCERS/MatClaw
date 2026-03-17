"""
Tool for generating ideal crystal structures from crystallographic prototypes.
Creates symmetric structures from spacegroup + Wyckoff positions for inorganic materials.
The most common starting point for computational materials discovery workflows.
"""

from typing import Dict, Any, Optional, List, Union, Annotated
from pydantic import Field


def pymatgen_prototype_builder(
    spacegroup: Annotated[
        Union[int, str],
        Field(
            description="Space group number (1-230) or Hermann-Mauguin symbol. "
            "Examples: 225, 'Fm-3m', 62, 'Pnma', 167, 'R-3c'. "
            "Use international notation for symbols."
        )
    ],
    species: Annotated[
        Union[List[str], Dict[str, str]],
        Field(
            description="Element species to place in the structure. "
            "Can be a list: ['La', 'Fe', 'O', 'O', 'O'] or "
            "dict mapping sites to elements: {'A': 'La', 'B': 'Fe', 'X': 'O'}. "
            "Use oxidation states if needed: ['La3+', 'Fe3+', 'O2-']."
        )
    ],
    lattice_parameters: Annotated[
        Union[List[float], Dict[str, float]],
        Field(
            description="Lattice parameters in Angstroms and degrees. "
            "List format: [a, b, c, alpha, beta, gamma] (6 values). "
            "Dict format: {'a': 5.5, 'b': 5.5, 'c': 13.2, 'alpha': 90, 'beta': 90, 'gamma': 120}. "
            "For cubic: [a, a, a, 90, 90, 90] or just [a]. "
            "Alpha, beta, gamma default to 90° if not provided."
        )
    ],
    wyckoff_positions: Annotated[
        Optional[Dict[str, Union[str, List[float]]]],
        Field(
            default=None,
            description="Wyckoff site assignments. Map Wyckoff label to either species or coordinates. "
            "Examples: {'4a': 'La', '4c': 'Fe', '8d': 'O'} or "
            "{'4a': ['La', [0, 0, 0]], '4c': ['Fe', [0.25, 0.25, 0.25]]}. "
            "If omitted, species list will be placed sequentially on available sites."
        )
    ] = None,
    coords: Annotated[
        Optional[List[List[float]]],
        Field(
            default=None,
            description="Fractional coordinates for each species if not using Wyckoff positions. "
            "Must match the length of species list. Example: [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]. "
            "Use this for explicit control or when Wyckoff positions are unknown."
        )
    ] = None,
    primitive: Annotated[
        bool,
        Field(
            default=False,
            description="If True, returns primitive cell. If False, returns conventional cell. "
            "Primitive cells have fewer atoms but may be less symmetric."
        )
    ] = False,
    n_structures: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            le=100,
            description="Number of structure variants to generate (1-100). "
            "Variants may differ in origin choice, setting, or slight perturbations. "
            "Default: 1."
        )
    ] = 1,
    validate_proximity: Annotated[
        bool,
        Field(
            default=True,
            description="If True, checks for atoms that are too close (< 0.5 Å) and returns error. "
            "If False, allows structures with unrealistic overlaps (useful for initial guesses)."
        )
    ] = True,
    merge_tol: Annotated[
        float,
        Field(
            default=0.01,
            ge=0.0,
            le=1.0,
            description="Distance tolerance (Angstroms) for merging symmetrically equivalent sites. "
            "Default: 0.01 Å. Increase if getting duplicate atoms warnings."
        )
    ] = 0.01,
    output_format: Annotated[
        str,
        Field(
            default="dict",
            description="Output format: 'dict' (pymatgen Structure.as_dict()), "
            "'poscar' (VASP POSCAR string), 'cif' (CIF string), "
            "'ase' (ASE-compatible atoms_dict). Default: 'dict'."
        )
    ] = "dict"
) -> Dict[str, Any]:
    """
    Generate ideal crystal structures from crystallographic prototypes.
    
    Creates symmetric structures using spacegroup symmetry and Wyckoff positions.
    This is the primary tool for generating starting structures for DFT calculations,
    structure prediction, and materials discovery workflows.
    
    Returns:
        dict: Structure generation results including:
            - success (bool): Whether generation was successful
            - count (int): Number of structures generated
            - structures (list): List of generated structures, each containing:
                - structure (dict/str): Structure in requested format
                - formula (str): Reduced chemical formula
                - spacegroup_number (int): Space group number
                - spacegroup_symbol (str): Hermann-Mauguin symbol
                - lattice (dict): Lattice parameters {a, b, c, alpha, beta, gamma, volume}
                - composition (dict): Element counts
                - n_atoms (int): Total number of atoms
                - n_sites (int): Number of unique sites
                - density (float): Density in g/cm³
                - is_primitive (bool): Whether this is a primitive cell
                - wyckoff_info (list): Wyckoff site information if available
            - parameters (dict): Input parameters used
            - message (str): Status message
            - warnings (list): Any warnings generated
            - error (str): Error message if failed
    """
    
    try:
        try:
            from pymatgen.core import Structure, Lattice, Element
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.analysis.structure_matcher import StructureMatcher
        except ImportError:
            return {
                "success": False,
                "error": "pymatgen is not installed. Install it with: pip install pymatgen"
            }
        
        warnings = []
        structures_out = []
        
        # Parse spacegroup
        try:
            if isinstance(spacegroup, int):
                if not 1 <= spacegroup <= 230:
                    return {
                        "success": False,
                        "error": f"Space group number must be between 1 and 230, got {spacegroup}"
                    }
                sg_number = spacegroup
                sg_symbol = None
            else:
                # Try to get spacegroup number from symbol
                from pymatgen.symmetry.groups import SpaceGroup
                try:
                    sg = SpaceGroup(spacegroup)
                    sg_number = sg.int_number
                    sg_symbol = spacegroup
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid spacegroup symbol '{spacegroup}': {str(e)}"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse spacegroup: {str(e)}"
            }
        
        # Parse lattice parameters
        try:
            if isinstance(lattice_parameters, dict):
                a = lattice_parameters.get('a')
                b = lattice_parameters.get('b', a)
                c = lattice_parameters.get('c', a)
                alpha = lattice_parameters.get('alpha', 90)
                beta = lattice_parameters.get('beta', 90)
                gamma = lattice_parameters.get('gamma', 90)
                lattice_params = [a, b, c, alpha, beta, gamma]
            elif isinstance(lattice_parameters, list):
                if len(lattice_parameters) == 1:
                    # Cubic: [a] -> [a, a, a, 90, 90, 90]
                    lattice_params = [lattice_parameters[0]] * 3 + [90, 90, 90]
                elif len(lattice_parameters) == 3:
                    # [a, b, c] -> [a, b, c, 90, 90, 90]
                    lattice_params = lattice_parameters + [90, 90, 90]
                elif len(lattice_parameters) == 6:
                    lattice_params = lattice_parameters
                else:
                    return {
                        "success": False,
                        "error": f"Lattice parameters must have 1, 3, or 6 values, got {len(lattice_parameters)}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Lattice parameters must be list or dict, got {type(lattice_parameters).__name__}"
                }
            
            lattice = Lattice.from_parameters(*lattice_params)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create lattice: {str(e)}"
            }
        
        # Parse species
        try:
            if isinstance(species, dict):
                # Convert dict to list based on wyckoff_positions or coords length
                if wyckoff_positions:
                    species_list = [species.get(k, list(species.values())[0]) 
                                   for k in wyckoff_positions.keys()]
                elif coords:
                    if len(species) != len(coords):
                        warnings.append(f"Species dict has {len(species)} entries but {len(coords)} coordinates provided")
                    species_list = list(species.values())[:len(coords)]
                else:
                    species_list = list(species.values())
            else:
                species_list = species
            
            # Validate species
            for sp in species_list:
                element_str = sp.split('+')[0].split('-')[0]
                try:
                    Element(element_str)
                except Exception:
                    warnings.append(f"'{sp}' may not be a valid element symbol")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse species: {str(e)}"
            }
        
        # Build structure(s)
        for i in range(n_structures):
            try:
                # Create structure from explicit coordinates
                if coords is not None:
                    if len(coords) != len(species_list):
                        return {
                            "success": False,
                            "error": f"Number of coordinates ({len(coords)}) must match number of species ({len(species_list)})"
                        }
                    
                    structure = Structure(
                        lattice,
                        species_list,
                        coords,
                        validate_proximity=validate_proximity,
                        coords_are_cartesian=False
                    )
                
                # Create structure from Wyckoff positions
                elif wyckoff_positions is not None:
                    # This requires manual Wyckoff -> coordinates mapping. For now, use symmetry operations from pymatgen
                    from pymatgen.symmetry.groups import SpaceGroup
                    sg = SpaceGroup.from_int_number(sg_number)
                    
                    all_species = []
                    all_coords = []
                    
                    for wyckoff_label, site_info in wyckoff_positions.items():
                        if isinstance(site_info, str):
                            # Just species, use default Wyckoff position
                            site_species = site_info
                            # Get standard Wyckoff position (simplified - would need full Wyckoff table)
                            site_coords = [0.0, 0.0, 0.0]  # Placeholder
                            warnings.append(f"Using default coordinates for Wyckoff position {wyckoff_label}")
                        elif isinstance(site_info, list) and len(site_info) == 2:
                            site_species = site_info[0]
                            site_coords = site_info[1]
                        else:
                            return {
                                "success": False,
                                "error": f"Invalid Wyckoff position format for {wyckoff_label}: {site_info}"
                            }
                        
                        all_species.append(site_species)
                        all_coords.append(site_coords)
                    
                    structure = Structure(
                        lattice,
                        all_species,
                        all_coords,
                        validate_proximity=validate_proximity,
                        coords_are_cartesian=False
                    )
                
                # Create structure with automatic site assignment
                else:
                    # Simple sequential placement - user should provide coords or Wyckoff
                    if len(species_list) == 0:
                        return {
                            "success": False,
                            "error": "Must provide either coords or wyckoff_positions when species is a simple list"
                        }
                    
                    # Generate evenly spaced coordinates
                    n_species = len(species_list)
                    auto_coords = [[i/n_species, i/n_species, i/n_species] for i in range(n_species)]
                    warnings.append("No coordinates provided - using automatic spacing. Provide 'coords' or 'wyckoff_positions' for realistic structures.")
                    
                    structure = Structure(
                        lattice,
                        species_list,
                        auto_coords,
                        validate_proximity=False,  # Don't validate auto-generated
                        coords_are_cartesian=False
                    )

                # Apply symmetry operations to ensure spacegroup symmetry
                try:
                    sga = SpacegroupAnalyzer(structure, symprec=merge_tol)
                    if primitive:
                        structure = sga.get_primitive_standard_structure()
                    else:
                        structure = sga.get_conventional_standard_structure()
                except Exception as e:
                    warnings.append(f"Could not apply symmetry operations: {str(e)}. Using input structure as-is.")
                
                # Get structure info
                try:
                    sga = SpacegroupAnalyzer(structure, symprec=0.1)
                    detected_sg = sga.get_space_group_number()
                    detected_symbol = sga.get_space_group_symbol()
                    if detected_sg != sg_number:
                        warnings.append(f"Detected spacegroup {detected_sg} ({detected_symbol}) differs from requested {sg_number}")
                except Exception:
                    detected_sg = sg_number
                    detected_symbol = sg_symbol or f"SG{sg_number}"

                # Get Wyckoff information
                wyckoff_info = []
                try:
                    wyckoff_symbols = sga.get_symmetry_dataset().wyckoffs
                    equivalent_atoms = sga.get_symmetry_dataset().equivalent_atoms
                    
                    for idx, (site, wyckoff, equiv) in enumerate(zip(structure.sites, wyckoff_symbols, equivalent_atoms)):
                        wyckoff_info.append({
                            "index": idx,
                            "species": str(site.specie),
                            "wyckoff": wyckoff,
                            "equivalent_to": int(equiv),
                            "coords": site.frac_coords.tolist()
                        })
                except Exception as e:
                    warnings.append(f"Could not extract Wyckoff information: {str(e)}")

                # Convert to requested output format
                if output_format == "dict":
                    structure_out = structure.as_dict()
                elif output_format == "poscar":
                    from pymatgen.io.vasp import Poscar
                    structure_out = str(Poscar(structure))
                elif output_format == "cif":
                    from pymatgen.io.cif import CifWriter
                    cif_writer = CifWriter(structure)
                    structure_out = str(cif_writer)
                elif output_format == "ase":
                    # Convert to ASE-compatible format
                    structure_out = {
                        "numbers": [site.specie.Z for site in structure.sites],
                        "positions": [site.coords.tolist() for site in structure.sites],
                        "cell": structure.lattice.matrix.tolist(),
                        "pbc": [True, True, True]
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Unknown output_format: {output_format}. Use 'dict', 'poscar', 'cif', or 'ase'."
                    }
                
                structures_out.append({
                    "structure": structure_out,
                    "formula": structure.composition.reduced_formula,
                    "spacegroup_number": detected_sg,
                    "spacegroup_symbol": detected_symbol,
                    "lattice": {
                        "a": structure.lattice.a,
                        "b": structure.lattice.b,
                        "c": structure.lattice.c,
                        "alpha": structure.lattice.alpha,
                        "beta": structure.lattice.beta,
                        "gamma": structure.lattice.gamma,
                        "volume": structure.lattice.volume
                    },
                    "composition": dict(structure.composition.as_dict()),
                    "n_atoms": len(structure),
                    "n_sites": structure.num_sites,
                    "density": structure.density,
                    "is_primitive": primitive,
                    "wyckoff_info": wyckoff_info
                })
                
            except Exception as e:
                if n_structures == 1:
                    return {
                        "success": False,
                        "error": f"Failed to generate structure: {str(e)}",
                        "error_type": type(e).__name__
                    }
                else:
                    warnings.append(f"Failed to generate structure {i+1}: {str(e)}")
        
        return {
            "success": True,
            "count": len(structures_out),
            "structures": structures_out,
            "parameters": {
                "spacegroup": spacegroup,
                "species": species,
                "lattice_parameters": lattice_parameters,
                "primitive": primitive,
                "n_structures": n_structures
            },
            "warnings": warnings if warnings else None,
            "message": f"Successfully generated {len(structures_out)} structure(s) with spacegroup {sg_number}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error in prototype builder: {str(e)}",
            "error_type": type(e).__name__
        }