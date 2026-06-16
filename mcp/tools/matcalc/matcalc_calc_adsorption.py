"""
Tool for calculating adsorption energies using matcalc.

Computes adsorption energy by comparing the energy of an adsorbate-slab system
to the clean slab and isolated adsorbate. Can automatically place adsorbates on
slabs at common adsorption sites (ontop, bridge, hollow).

Use this tool to:
- Calculate adsorption energy for molecules on surfaces
- Screen different adsorbates on a catalyst surface
- Find optimal adsorption sites
- Compare adsorption strength across different surfaces
"""

from typing import Dict, Any, Optional, Union, Annotated, Tuple, List
from pydantic import Field

import matcalc as mtc
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def matcalc_calc_adsorption(
    clean_slab_structure: Annotated[
        str,
        Field(
            description=(
                "Clean slab structure as a CIF or POSCAR string. "
                "Should already be a slab with vacuum, or use "
                "matcalc_calc_surface to generate one from bulk. The adsorbate will be "
                "placed on this slab surface in generated-site mode."
            )
        )
    ],
    adsorbate: Annotated[
        Union[str, List[float]],
        Field(
            description=(
                "Adsorbate identity or structure. In generated-site mode this is placed on "
                "the clean slab. In custom mode this is used to validate the extracted "
                "adsorbate and define the isolated adsorbate reference. Can be:\n"
                "- String: Molecular formula like 'CO', 'H2O', 'CH4', 'O', 'H' (will use pymatgen to build)\n"
                "- List of floats [x, y, z]: Single atom position (will place at this height above surface)"
            )
        )
    ],
    adslab_structure: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Optional adsorbate-added slab structure as a CIF or POSCAR string. "
                "Provide this together with adsorbate_indices to use a custom adsorption "
                "geometry instead of generating one from adsorption_site. When using "
                "adsorbate_indices, POSCAR is preferred because it preserves atom ordering "
                "more reliably than CIF round-trips."
            )
        )
    ] = None,
    adsorbate_indices: Annotated[
        Optional[List[int]],
        Field(
            default=None,
            description=(
                "Optional zero-based atom indices in adslab_structure that belong to the adsorbate. "
                "Required when adslab_structure is provided. Indices are interpreted after "
                "parsing the supplied structure, so use POSCAR if you need stable ordering."
            )
        )
    ] = None,
    adsorption_site: Annotated[
        str,
        Field(
            default="ontop",
            description=(
                "Type of adsorption site to use. Options:\n"
                "- 'ontop': Directly above a surface atom\n"
                "- 'bridge': Between two surface atoms\n"
                "- 'hollow': In the center of 3+ surface atoms\n"
                "- 'all': Try all sites and return best (lowest energy)"
            )
        )
    ] = "ontop",
    distance: Annotated[
        float,
        Field(
            default=2.0,
            ge=1.0,
            le=4.0,
            description=(
                "Distance from adsorbate to surface in Angstroms (1-4 Å). "
                "Default: 2.0 Å."
            )
        )
    ] = 2.0,
    calculator: Annotated[
        str,
        Field(
            default="TensorNet-PES-MatPES-r2SCAN-2025.2",
            description=(
                "Foundation ML potential name. Options include "
                "'MACE-MPA-0-medium', 'MACE-MP-0-medium', 'MACE-OMAT-0-medium', 'MACE-MatPES-r2SCAN-0', "
                "'TensorNet-MatPES-r2SCAN-2025.2', 'CHGNet-MatPES-PBE-2025.2.10', 'M3GNet-MatPES-PBE-2025.1', "
                "'SevenNet-0', 'ORB-v2', 'MatterSim-v1.0.0-5M'. "
                "Default: TensorNet-PES-MatPES-r2SCAN-2025.2."
            )
        )
    ] = "TensorNet-PES-MatPES-r2SCAN-2025.2",
    relax_adsorbate: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True (default), relaxes the isolated adsorbate before calculating energy. "
                "Set to False if adsorbate geometry is already optimized."
            )
        )
    ] = True,
    relax_slab: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "If True (default), relaxes the clean slab before calculating energy. "
                "Set to False if slab is already at equilibrium."
            )
        )
    ] = True,
    relax_bulk: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If True, relaxes the bulk structure (if needed). Usually not required "
                "for adsorption calculations. Default: False."
            )
        )
    ] = False,
    fmax: Annotated[
        float,
        Field(
            default=0.1,
            ge=0.01,
            le=1.0,
            description=(
                "Force convergence tolerance in eV/Å for structure relaxation (0.01-1.0). "
                "Lower values = more accurate but slower. Default: 0.1 eV/Å."
            )
        )
    ] = 0.1,
    optimizer: Annotated[
        str,
        Field(
            default="BFGS",
            description=(
                "Optimization algorithm for relaxation. Options:\n"
                "- 'BFGS' (default, Quasi-Newton method)\n"
                "- 'FIRE' (Fast Inertial Relaxation Engine)\n"
                "- 'LBFGS' (Limited-memory BFGS)\n"
                "- 'BFGSLineSearch' (BFGS with line search)"
            )
        )
    ] = "BFGS",
    max_steps: Annotated[
        int,
        Field(
            default=500,
            ge=10,
            le=2000,
            description=(
                "Maximum optimization steps (10-2000). Default: 500."
            )
        )
    ] = 500,
) -> Dict[str, Any]:
    """
    Calculate adsorption energy for a molecule/atom on a surface.
    
    Places the adsorbate on the slab at the specified site type, relaxes the
    system, and computes the adsorption energy as:
    E_ads = E_adslab - E_slab - E_adsorbate
    
    Args:
        clean_slab_structure: Clean slab surface structure (CIF/POSCAR string)
        adslab_structure: Optional user-supplied adsorbate+slab structure
        adsorbate: Molecule/atom to adsorb (formula, coords, or Molecule dict)
        adsorbate_indices: Optional zero-based atom indices identifying the adsorbate in adslab_structure
        adsorption_site: Type of site ('ontop', 'bridge', 'hollow', 'all')
        distance: Initial adsorbate-surface distance in Angstroms
        calculator: Force field or ML potential to use
        relax_adsorbate: Whether to relax isolated adsorbate
        relax_slab: Whether to relax clean slab
        relax_bulk: Whether to relax bulk (usually not needed)
        fmax: Force convergence tolerance in eV/Å
        optimizer: Optimization algorithm
        max_steps: Maximum optimization steps
        
    Returns:
        Dictionary containing:
        - adsorption_energy: Adsorption energy in eV (negative = favorable)
        - adslab_energy: Total energy of adsorbate+slab system in eV
        - slab_energy: Energy of clean slab in eV
        - adsorbate_energy: Energy of isolated adsorbate in eV
        - slab_energy_per_atom: Slab energy per atom in eV/atom
        - adslab_structure: Final adslab structure as CIF string
        - slab_structure: Final slab structure as CIF string
        - adsorbate_structure: Final adsorbate structure as XYZ string
        - adsorption_site: Site type used, or 'custom' when adslab_structure is supplied
        - adsorption_mode: 'generated' or 'custom'
        - adsorbate_indices: Adsorbate indices used in custom mode
        - num_slab_atoms: Number of atoms in slab
        - num_adsorbate_atoms: Number of atoms in adsorbate
        
    Raises:
        ValueError: If structure/adsorbate parsing fails or site is invalid
        RuntimeError: If adsorption calculation fails
    """
    
    # Parse clean slab structure
    try:
        slab = _parse_structure(clean_slab_structure)
    except Exception as e:
        return {
            "error": f"Failed to parse clean slab structure: {str(e)}",
            "details": "Ensure clean_slab_structure is a valid CIF or POSCAR string."
        }

    custom_mode = adslab_structure is not None or adsorbate_indices is not None
    if custom_mode and (adslab_structure is None or adsorbate_indices is None):
        return {
            "error": "Custom adsorption mode requires both adslab_structure and adsorbate_indices",
            "details": "Provide adslab_structure together with zero-based adsorbate_indices, or omit both to use adsorption_site generation."
        }

    # Parse adsorbate reference
    try:
        adsorbate_ref = _parse_adsorbate(adsorbate)
    except Exception as e:
        return {
            "error": f"Failed to parse adsorbate: {str(e)}",
            "details": "Ensure adsorbate is a molecular formula, coords, or Molecule dict."
        }

    adsorption_mode = "custom" if custom_mode else "generated"
    adsorption_site_used = "custom" if custom_mode else adsorption_site

    if custom_mode:
        try:
            adslab = _parse_structure(adslab_structure)
        except Exception as e:
            return {
                "error": f"Failed to parse adslab structure: {str(e)}",
                "details": "Ensure adslab_structure is a valid CIF or POSCAR string.",
                "adsorption_mode": adsorption_mode,
            }

        try:
            adsorbate_mol = _extract_adsorbate_from_structure(adslab, adsorbate_indices)
        except Exception as e:
            return {
                "error": f"Failed to extract adsorbate from adslab structure: {str(e)}",
                "details": "Ensure adsorbate_indices are valid zero-based atom indices in adslab_structure.",
                "adsorption_mode": adsorption_mode,
            }

        if adsorbate_mol.composition.alphabetical_formula != adsorbate_ref.composition.alphabetical_formula:
            return {
                "error": "Adsorbate identity does not match extracted adsorbate indices",
                "details": (
                    f"Extracted adsorbate formula is {adsorbate_mol.composition.alphabetical_formula}, "
                    f"but parsed adsorbate formula is {adsorbate_ref.composition.alphabetical_formula}."
                ),
                "adsorption_mode": adsorption_mode,
                "adsorbate_indices": adsorbate_indices,
            }
    else:
        adsorbate_mol = adsorbate_ref

        # Find adsorption sites and place adsorbate
        try:
            asf = AdsorbateSiteFinder(slab)

            if adsorption_site == "all":
                ads_structs = asf.generate_adsorption_structures(
                    adsorbate_mol,
                    repeat=[1, 1, 1],
                    find_args={'distance': distance}
                )
            else:
                ads_structs = asf.generate_adsorption_structures(
                    adsorbate_mol,
                    repeat=[1, 1, 1],
                    find_args={'distance': distance}
                )
                filtered = []
                for ads_struct in ads_structs:
                    site_props = getattr(ads_struct, 'properties', {})
                    site_name = site_props.get('adsorption_site', '').lower()
                    if adsorption_site.lower() in site_name or not site_name:
                        filtered.append(ads_struct)
                ads_structs = filtered if filtered else ads_structs

            if not ads_structs:
                return {
                    "error": "No valid adsorption structures generated",
                    "details": f"Could not place adsorbate at '{adsorption_site}' sites with distance={distance} Å",
                    "adsorption_site": adsorption_site,
                    "adsorption_mode": adsorption_mode,
                }

            adslab = ads_structs[0]

        except Exception as e:
            return {
                "error": f"Failed to generate adsorption structure: {str(e)}",
                "details": "Could not place adsorbate on slab surface.",
                "adsorption_site": adsorption_site,
                "adsorption_mode": adsorption_mode,
            }
    
    # Load calculator
    try:
        calc = mtc.load_fp(calculator)
    except Exception as e:
        return {
            "error": f"Failed to load calculator '{calculator}': {str(e)}",
            "details": "Valid names are the keys of matcalc.utils.MODEL_REGISTRY (and MODEL_ALIASES for short names), e.g. 'MACE-MPA-0-medium' / alias 'mace' — MACE names additionally require the 'mace-torch' package."
        }
    
    # Create AdsorptionCalc
    try:
        adsorption_calc = mtc.AdsorptionCalc(
            calculator=calc,
            relax_adsorbate=relax_adsorbate,
            relax_slab=relax_slab,
            relax_bulk=relax_bulk,
            fmax=fmax,
            optimizer=optimizer,
            max_steps=max_steps,
        )
    except Exception as e:
        return {
            "error": f"Failed to initialize AdsorptionCalc: {str(e)}",
            "details": "Check optimizer and parameter values."
        }
    
    # Run adsorption calculation
    try:
        # AdsorptionCalc requires dict with adslab, slab, adsorbate
        adsorption_input = {
            'adslab': adslab,
            'slab': slab,
            'adsorbate': adsorbate_mol
        }
        results = adsorption_calc.calc(adsorption_input)
    except Exception as e:
        return {
            "error": f"Adsorption calculation failed: {str(e)}",
            "details": "Energy calculation encountered an error.",
            "adsorption_site": adsorption_site
        }
    
    # Format output
    try:
        from pymatgen.io.cif import CifWriter
        from pymatgen.io.xyz import XYZ
        
        output = {
            "adsorption_energy": float(results["adsorption_energy"]),
            "adsorption_energy_units": "eV",
            "adslab_energy": float(results["adslab_energy"]),
            "slab_energy": float(results["slab_energy"]),
            "adsorbate_energy": float(results["adsorbate_energy"]),
            "slab_energy_per_atom": float(results["slab_energy_per_atom"]),
            "energy_units": "eV",
            "adslab_structure": str(CifWriter(results["final_adslab"])),
            "clean_slab_structure": str(CifWriter(results["final_slab"])),
            "adsorbate_structure": str(XYZ(results["final_adsorbate"])),
            "adsorption_site": adsorption_site_used,
            "adsorption_mode": adsorption_mode,
            "adsorbate_indices": adsorbate_indices,
            "distance": distance,
            "num_slab_atoms": len(slab),
            "num_adsorbate_atoms": len(adsorbate_mol),
            "num_adslab_atoms": len(adslab),
            "calculator": calculator,
            "relax_adsorbate": relax_adsorbate,
            "relax_slab": relax_slab,
            "relax_bulk": relax_bulk,
        }
        
        # Add interpretation
        if output["adsorption_energy"] < 0:
            output["adsorption_favorable"] = True
            output["interpretation"] = "Negative adsorption energy indicates favorable (exothermic) adsorption"
        else:
            output["adsorption_favorable"] = False
            output["interpretation"] = "Positive adsorption energy indicates unfavorable (endothermic) adsorption"
        
        return output
        
    except Exception as e:
        return {
            "error": f"Failed to format results: {str(e)}",
            "details": "Adsorption calculation completed but result formatting failed."
        }


def _parse_structure(structure_input: str) -> Structure:
    """
    Parse structure from string format.
    
    Args:
        structure_input: Structure as CIF/POSCAR string
        
    Returns:
        pymatgen Structure object
        
    Raises:
        ValueError: If parsing fails
    """
    if isinstance(structure_input, str):
        # Try CIF first, then POSCAR
        for fmt in ['cif', 'poscar']:
            try:
                return Structure.from_str(structure_input, fmt=fmt)
            except:
                continue
        raise ValueError("Could not parse structure string as CIF or POSCAR")
        
    else:
        raise ValueError(f"structure_input must be dict or str, got {type(structure_input)}")


def _parse_adsorbate(adsorbate: Union[str, List[float], Dict[str, Any]]) -> Molecule:
    """
    Parse adsorbate from various formats.
    
    Args:
        adsorbate: Molecular formula, coords, or Molecule dict
        
    Returns:
        pymatgen Molecule object
        
    Raises:
        ValueError: If parsing fails
    """
    if isinstance(adsorbate, str):
        # Treat as molecular formula
        try:
            # Use pymatgen's molecule building
            return Molecule.from_str(adsorbate, fmt='xyz')
        except:
            # Try as simple atom or molecule
            # Common adsorbates with reasonable geometries
            adsorbate_upper = adsorbate.upper()
            if adsorbate_upper == "CO":
                return Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.128]])
            elif adsorbate_upper == "O":
                return Molecule(["O"], [[0, 0, 0]])
            elif adsorbate_upper == "H":
                return Molecule(["H"], [[0, 0, 0]])
            elif adsorbate_upper == "OH":
                return Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]])
            elif adsorbate_upper == "H2O":
                return Molecule(["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
            elif adsorbate_upper == "CH4":
                return Molecule(["C", "H", "H", "H", "H"], 
                              [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
                               [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]])
            elif adsorbate_upper == "N2":
                return Molecule(["N", "N"], [[0, 0, 0], [0, 0, 1.098]])
            elif adsorbate_upper == "NO":
                return Molecule(["N", "O"], [[0, 0, 0], [0, 0, 1.151]])
            else:
                # Try as single atom
                return Molecule([adsorbate], [[0, 0, 0]])
                
    elif isinstance(adsorbate, list):
        # Treat as coords for single atom (will use 'X' placeholder)
        if len(adsorbate) == 3:
            # Assume it's just coords, use as position
            raise ValueError("Adsorbate as list not yet supported. Use molecular formula.")
        else:
            raise ValueError("Adsorbate list must have 3 coordinates [x, y, z]")
            
    else:
        raise ValueError(f"adsorbate must be str or list, got {type(adsorbate)}")


def _extract_adsorbate_from_structure(adslab: Structure, adsorbate_indices: List[int]) -> Molecule:
    """
    Build an adsorbate molecule from explicit indices in an adsorbate+slab structure.

    Args:
        adslab: Adsorbate+slab structure
        adsorbate_indices: Zero-based indices belonging to the adsorbate

    Returns:
        pymatgen Molecule object using the extracted cartesian coordinates

    Raises:
        ValueError: If indices are invalid or empty
    """
    if not adsorbate_indices:
        raise ValueError("adsorbate_indices cannot be empty")

    if len(set(adsorbate_indices)) != len(adsorbate_indices):
        raise ValueError("adsorbate_indices must be unique")

    num_sites = len(adslab)
    invalid_indices = [index for index in adsorbate_indices if index < 0 or index >= num_sites]
    if invalid_indices:
        raise ValueError(f"indices out of range for structure with {num_sites} sites: {invalid_indices}")

    sorted_indices = sorted(adsorbate_indices)
    species = [str(adslab[index].specie) for index in sorted_indices]
    coords = [adslab[index].coords for index in sorted_indices]
    return Molecule(species, coords)

