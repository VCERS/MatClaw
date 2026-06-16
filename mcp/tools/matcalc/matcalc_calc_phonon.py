"""
Phonon calculation tool using matcalc PhononCalc.

This tool calculates phonon properties and thermodynamic quantities using 
universal ML potentials (e.g., TensorNet-MatPES-PBE, M3GNet, CHGNet).
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from pymatgen.core import Structure


@contextmanager
def _temporary_working_directory():
    """Context manager to run phonopy calculations in a temporary directory.
    
    This prevents phonopy from writing intermediate files (phonon.yaml, FORCE_SETS, etc.)
    to the user's current working directory.
    """
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            os.chdir(temp_dir)
            yield temp_dir
        finally:
            os.chdir(original_dir)


def matcalc_calc_phonon(
    structure_input: Annotated[
        str,
        Field(
            description=(
                "Structure as CIF string, POSCAR string, dict, or pymatgen Structure. "
                "Can be: (1) CIF format string (must start with 'data_' or contain '_cell_'), "
                "(2) POSCAR format string, (3) Dictionary with structure data, "
                "(4) Pymatgen Structure object."
            )
        )
    ],
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
    atom_disp: Annotated[
        float,
        Field(
            default=0.015,
            gt=0.0,
            description=(
                "Atomic displacement distance for calculating force constants in Angstroms. "
                "Smaller values increase accuracy but may be numerically less stable. Default: 0.015."
            )
        )
    ] = 0.015,
    supercell_matrix: Annotated[
        list[list[int]] | None,
        Field(
            default=None,
            description=(
                "Supercell matrix for phonon calculations. Larger supercells give more accurate results "
                "but are more expensive. Can be: (1) List of 3 integers [a, b, c] for diagonal supercell, "
                "(2) 3x3 matrix [[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]]. "
                "Default: [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2×2×2 supercell)."
            )
        )
    ] = None,
    t_min: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Minimum temperature for thermodynamic properties in Kelvin. Default: 0.0."
        )
    ] = 0.0,
    t_max: Annotated[
        float,
        Field(
            default=1000.0,
            gt=0.0,
            description="Maximum temperature for thermodynamic properties in Kelvin. Default: 1000.0."
        )
    ] = 1000.0,
    t_step: Annotated[
        float,
        Field(
            default=10.0,
            gt=0.0,
            description="Temperature step for thermodynamic properties in Kelvin. Default: 10.0."
        )
    ] = 10.0,
    relax_structure: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether to relax the structure before phonon calculation. "
                "Recommended: True (ensuring structure is at equilibrium improves accuracy). Default: True."
            )
        )
    ] = True,
    fmax: Annotated[
        float,
        Field(
            default=0.1,
            gt=0.0,
            description=(
                "Force convergence criterion for structure relaxation in eV/Angstrom. "
                "Only used if relax_structure=True. Default: 0.1."
            )
        )
    ] = 0.1,
    **kwargs,
) -> dict[str, Any]:
    """
    Calculate phonon properties and thermodynamic quantities using universal ML potentials.
    """
    try:
        from matcalc import PhononCalc
        import matcalc as mtc
    except ImportError as err:
        return {
            "success": False,
            "error": f"Failed to import matcalc: {err}. Please install with: pip install matcalc",
        }

    # Parse structure
    try:
        structure = _parse_structure(structure_input)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse structure: {e}",
        }

    # Set up supercell matrix
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    elif isinstance(supercell_matrix, list) and len(supercell_matrix) == 3 and isinstance(supercell_matrix[0], (int, float)):
        # Handle [a, b, c] format
        a, b, c = supercell_matrix
        supercell_matrix = [[a, 0, 0], [0, b, 0], [0, 0, c]]

    # Load calculator
    try:
        calc = mtc.load_fp(calculator)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load calculator '{calculator}': {e}",
            "details": "Valid names are the keys of matcalc.utils.MODEL_REGISTRY (and MODEL_ALIASES for short names), e.g. 'MACE-MPA-0-medium' / alias 'mace' — MACE names additionally require the 'mace-torch' package." 
        }

    # Set up PhononCalc
    try:
        phonon_calc = PhononCalc(
            calculator=calc,
            atom_disp=atom_disp,
            supercell_matrix=supercell_matrix,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            relax_structure=relax_structure,
            fmax=fmax,
            **kwargs,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize PhononCalc: {e}",
        }

    # Run phonon calculation in temporary directory to avoid file pollution
    try:
        with _temporary_working_directory():
            result = phonon_calc.calc(structure)
    except Exception as e:
        return {
            "success": False,
            "error": f"Phonon calculation failed: {e}",
        }

    # Extract phonopy object and thermal properties
    phonon = result.get("phonon")
    thermal_props = result.get("thermal_properties", {})
    
    if phonon is None:
        return {
            "success": False,
            "error": "Phonon calculation did not return a phonopy object",
        }

    # Analyze phonon stability (check for imaginary modes)
    stability_info = _analyze_phonon_stability(phonon)
    
    # Calculate Debye temperature
    debye_temp = _calculate_debye_temperature(phonon)
    
    # Format thermal properties
    formatted_thermal = _format_thermal_properties(thermal_props)
    
    # Get final structure
    final_structure = result.get("final_structure", structure)
    from pymatgen.io.cif import CifWriter
    
    return {
        "success": True,
        "thermal_properties": formatted_thermal,
        "stability": stability_info,
        "debye_temperature": debye_temp,
        "structure": str(CifWriter(final_structure)),
        "relaxed": relax_structure,
        "calculator": calculator,
        "units": {
            "temperature": "K",
            "free_energy": "kJ/mol",
            "entropy": "J/K/mol",
            "heat_capacity": "J/K/mol",
            "frequency": "THz",
            "debye_temperature": "K",
        },
    }


def _parse_structure(structure_input: str | Structure) -> Structure:
    """Parse structure from various input formats."""
    if isinstance(structure_input, Structure):
        return structure_input
    
    if isinstance(structure_input, str):
        structure_input = structure_input.strip()
        
        # Try CIF format
        if "data_" in structure_input or "_cell_" in structure_input.lower():
            try:
                return Structure.from_str(structure_input, fmt="cif")
            except Exception as e:
                raise ValueError(f"Could not parse CIF format: {e}")
        
        # Try POSCAR format
        try:
            return Structure.from_str(structure_input, fmt="poscar")
        except Exception as e:
            raise ValueError(f"Could not parse POSCAR format: {e}")
    
    raise ValueError(f"Unsupported structure input type: {type(structure_input)}")


def _analyze_phonon_stability(phonon) -> dict[str, Any]:
    """
    Analyze phonon for imaginary modes (negative frequencies).
    
    Imaginary phonon modes indicate structural instability.
    """
    try:
        # Get mesh frequencies
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict.get("frequencies")  # Shape: (num_qpoints, num_branches)
        
        if frequencies is None:
            return {
                "is_stable": None,
                "num_imaginary_modes": None,
                "max_imaginary_frequency": None,
                "note": "Could not extract frequencies from phonopy mesh"
            }
        
        frequencies = np.array(frequencies)  # THz
        
        # Negative frequencies indicate imaginary modes
        # Use small tolerance to avoid counting numerical noise
        tolerance = 0.1  # THz
        imaginary_freqs = frequencies[frequencies < -tolerance]
        
        num_imaginary = len(imaginary_freqs)
        max_imaginary = float(imaginary_freqs.min()) if num_imaginary > 0 else None
        
        return {
            "is_stable": num_imaginary == 0,
            "num_imaginary_modes": num_imaginary,
            "max_imaginary_frequency": max_imaginary,
        }
        
    except Exception as e:
        return {
            "is_stable": None,
            "num_imaginary_modes": None,
            "max_imaginary_frequency": None,
            "note": f"Failed to analyze stability: {e}"
        }


def _calculate_debye_temperature(phonon) -> float:
    """
    Calculate Debye temperature from phonon DOS.
    
    The Debye temperature is a characteristic temperature that represents
    the maximum phonon frequency in a simplified model.
    """
    try:
        # Get total DOS
        total_dos = phonon.get_total_dos()
        
        if total_dos is None:
            return None
        
        # Debye temperature can be estimated from phonopy
        # θ_D = ħω_D / k_B where ω_D is Debye frequency
        # Phonopy calculates this as part of thermal properties
        
        # Get thermal properties at lowest temperature
        thermal_props = phonon.get_thermal_properties_dict()
        
        # Alternatively, estimate from average frequency
        # For now, we'll use a simple approximation based on max frequency
        mesh_dict = phonon.get_mesh_dict()
        frequencies = np.array(mesh_dict.get("frequencies", []))  # THz
        
        if len(frequencies) == 0:
            return None
        
        # Remove imaginary (negative) frequencies
        real_freqs = frequencies[frequencies > 0]
        
        if len(real_freqs) == 0:
            return None
        
        # Debye cutoff frequency (use 90th percentile as approximation)
        freq_debye = np.percentile(real_freqs, 90)  # THz
        
        # Convert to Debye temperature
        # θ_D = (h * ν_D) / k_B
        # where h = 6.62607015e-34 J·s (Planck constant)
        #       k_B = 1.380649e-23 J/K (Boltzmann constant)
        #       ν_D in Hz
        
        h = 6.62607015e-34  # J·s
        k_B = 1.380649e-23  # J/K
        freq_debye_hz = freq_debye * 1e12  # THz to Hz
        
        debye_temp = (h * freq_debye_hz) / k_B  # K
        
        return round(float(debye_temp), 1)
        
    except Exception as e:
        return None


def _format_thermal_properties(thermal_props: dict[str, Any]) -> dict[str, Any]:
    """
    Format thermal properties for cleaner output.
    
    Rounds values and ensures consistent types.
    """
    formatted = {}
    
    if "temperatures" in thermal_props:
        formatted["temperatures"] = [round(float(t), 2) for t in thermal_props["temperatures"]]
    
    if "free_energy" in thermal_props:
        formatted["free_energy"] = [round(float(f), 4) for f in thermal_props["free_energy"]]
    
    if "entropy" in thermal_props:
        formatted["entropy"] = [round(float(s), 4) for s in thermal_props["entropy"]]
    
    if "heat_capacity" in thermal_props:
        formatted["heat_capacity"] = [round(float(cv), 4) for cv in thermal_props["heat_capacity"]]
    
    return formatted


# For testing
if __name__ == "__main__":
    # Simple test with Si structure
    si_poscar = """Si2
1.0
3.348920 0.000000 1.933487
1.116307 3.157372 1.933487
0.000000 0.000000 3.866975
Si
2
direct
0.875000 0.875000 0.875000 Si
0.125000 0.125000 0.125000 Si"""
    
    result = matcalc_calc_phonon(
        structure_input=si_poscar,
        calculator="M3GNet",
        supercell_matrix=[2, 2, 2],
        t_max=500.0,
        relax_structure=False,
    )
    
    print("Success:", result.get("success"))
    if result.get("success"):
        print(f"Debye temperature: {result.get('debye_temperature')} K")
        print(f"Stable: {result['stability']['is_stable']}")
        print(f"Number of temperatures: {len(result['thermal_properties']['temperatures'])}")
    else:
        print("Error:", result.get("error"))
