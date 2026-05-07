"""
Thermal conductivity calculation tool using matcalc Phonon3Calc.

This tool calculates lattice thermal conductivity using third-order force constants
and the Boltzmann transport equation (BTE) within the relaxation time approximation (RTA).
Uses universal ML potentials (e.g., TensorNet-MatPES-PBE, M3GNet, CHGNet).
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


def matcalc_calc_phonon3(
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
                "Calculator/potential to use. "
                "For the full list of available calculators, run `matgl.get_available_pretrained_models`"
            )
        )
    ] = "TensorNet-PES-MatPES-r2SCAN-2025.2",
    fc2_supercell: Annotated[
        list[list[int]] | None,
        Field(
            default=None,
            description=(
                "Supercell matrix for second-order force constants (harmonic). "
                "Larger supercells give more accurate phonon properties. "
                "Can be: (1) List of 3 integers [a, b, c] for diagonal supercell, "
                "(2) 3x3 matrix [[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]]. "
                "Default: [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2×2×2 supercell)."
            )
        )
    ] = None,
    fc3_supercell: Annotated[
        list[list[int]] | None,
        Field(
            default=None,
            description=(
                "Supercell matrix for third-order force constants (anharmonic). "
                "Should typically match or exceed fc2_supercell for consistency. Same format as fc2_supercell. "
                "Default: [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2×2×2 supercell)."
            )
        )
    ] = None,
    mesh_numbers: Annotated[
        list[int] | None,
        Field(
            default=None,
            description=(
                "q-point mesh for thermal conductivity integration [nx, ny, nz]. "
                "Denser mesh = more accurate but more expensive. Default: [20, 20, 20]."
            )
        )
    ] = None,
    t_min: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            description="Minimum temperature for thermal conductivity calculation in Kelvin. Default: 0.0."
        )
    ] = 0.0,
    t_max: Annotated[
        float,
        Field(
            default=1000.0,
            gt=0.0,
            description="Maximum temperature for thermal conductivity calculation in Kelvin. Default: 1000.0."
        )
    ] = 1000.0,
    t_step: Annotated[
        float,
        Field(
            default=10.0,
            gt=0.0,
            description="Temperature step in Kelvin. Default: 10.0."
        )
    ] = 10.0,
    relax_structure: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether to relax the structure before calculation. "
                "Recommended: True (equilibrium structure needed for accurate force constants). Default: True."
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
    Calculate lattice thermal conductivity using third-order force constants and Boltzmann transport equation (RTA).
    """
    try:
        from matcalc import Phonon3Calc
        import matcalc as mtc
    except ImportError as err:
        return {
            "success": False,
            "error": f"Failed to import matcalc: {err}. Please install with: pip install matcalc",
        }
    
    # Workaround for phono3py 3.30.1+ compatibility
    # matcalc tries to access kappa_TOT_RTA but phono3py 3.30.1 uses just 'kappa'
    try:
        from phono3py.conductivity.rta_init import RTACalculator
        if not hasattr(RTACalculator, 'kappa_TOT_RTA'):
            # Add compatibility property
            @property
            def kappa_TOT_RTA(self):
                """Compatibility property for older matcalc versions."""
                return self.kappa
            RTACalculator.kappa_TOT_RTA = kappa_TOT_RTA
    except Exception:
        # If monkey-patch fails, continue anyway - might not be needed
        pass

    # Parse structure
    try:
        structure = _parse_structure(structure_input)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse structure: {e}",
        }

    # Set up supercell matrices
    if fc2_supercell is None:
        fc2_supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    elif isinstance(fc2_supercell, list) and len(fc2_supercell) == 3 and isinstance(fc2_supercell[0], (int, float)):
        # Handle [a, b, c] format
        a, b, c = fc2_supercell
        fc2_supercell = [[a, 0, 0], [0, b, 0], [0, 0, c]]
    
    if fc3_supercell is None:
        fc3_supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    elif isinstance(fc3_supercell, list) and len(fc3_supercell) == 3 and isinstance(fc3_supercell[0], (int, float)):
        # Handle [a, b, c] format
        a, b, c = fc3_supercell
        fc3_supercell = [[a, 0, 0], [0, b, 0], [0, 0, c]]
    
    if mesh_numbers is None:
        mesh_numbers = [20, 20, 20]

    # Load calculator
    try:
        calc = mtc.load_fp(calculator)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load calculator '{calculator}': {e}",
            "details": "Check that calculator is available using `matgl.get_available_pretrained_models()`"
        }

    # Extract optional kwargs
    disp_kwargs = kwargs.pop("disp_kwargs", {})
    thermal_conductivity_kwargs = kwargs.pop("thermal_conductivity_kwargs", {})
    optimizer = kwargs.pop("optimizer", "FIRE")
    write_phonon3 = kwargs.pop("write_phonon3", False)
    write_kappa = kwargs.pop("write_kappa", False)
    relax_calc_kwargs = kwargs.pop("relax_calc_kwargs", None)

    # Set up Phonon3Calc
    try:
        phonon3_calc = Phonon3Calc(
            calculator=calc,
            fc2_supercell=fc2_supercell,
            fc3_supercell=fc3_supercell,
            mesh_numbers=mesh_numbers,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            relax_structure=relax_structure,
            fmax=fmax,
            optimizer=optimizer,
            disp_kwargs=disp_kwargs,
            thermal_conductivity_kwargs=thermal_conductivity_kwargs,
            write_phonon3=write_phonon3,
            write_kappa=write_kappa,
            relax_calc_kwargs=relax_calc_kwargs,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize Phonon3Calc: {e}",
        }

    # Run thermal conductivity calculation in temporary directory to avoid file pollution
    try:
        with _temporary_working_directory():
            result = phonon3_calc.calc(structure)
    except Exception as e:
        return {
            "success": False,
            "error": f"Phonon3 calculation failed: {e}",
        }

    # Extract results
    temperatures = result.get("temperatures")
    kappa = result.get("thermal_conductivity")
    
    if temperatures is None or kappa is None:
        return {
            "success": False,
            "error": "Phonon3 calculation did not return thermal conductivity data",
        }

    # Format thermal conductivity
    kappa_formatted = _format_thermal_conductivity(kappa, temperatures)
    
    # Get final structure
    final_structure = result.get("final_structure", structure)
    
    from pymatgen.io.cif import CifWriter
    
    return {
        "success": True,
        "thermal_conductivity": kappa_formatted["kappa"],
        "temperatures": kappa_formatted["temperatures"],
        "structure": str(CifWriter(final_structure)),
        "relaxed": relax_structure,
        "calculator": calculator,
        "parameters": {
            "fc2_supercell": fc2_supercell,
            "fc3_supercell": fc3_supercell,
            "mesh_numbers": list(mesh_numbers),
            "t_min": t_min,
            "t_max": t_max,
            "t_step": t_step,
            "fmax": fmax,
        },
        "units": {
            "temperature": "K",
            "thermal_conductivity": "W/m·K",
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


def _format_thermal_conductivity(kappa: np.ndarray, temperatures: np.ndarray) -> dict[str, Any]:
    """
    Format thermal conductivity for cleaner output.
    
    Handles NaN values and ensures consistent types.
    """
    # Convert to lists and round values
    temps = [round(float(t), 2) for t in temperatures]
    
    # Handle scalar or array kappa
    if np.isscalar(kappa) or kappa.size == 1:
        # Single value
        k_val = float(kappa)
        kappa_list = [round(k_val, 4) if not np.isnan(k_val) else None]
    else:
        # Array of values
        kappa_list = []
        for k in kappa.flat:
            k_float = float(k)
            kappa_list.append(round(k_float, 4) if not np.isnan(k_float) else None)
    
    return {
        "temperatures": temps,
        "kappa": kappa_list,
    }


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
    
    result = matcalc_calc_phonon3(
        structure_input=si_poscar,
        calculator="M3GNet",
        fc2_supercell=[2, 2, 2],
        fc3_supercell=[2, 2, 2],
        mesh_numbers=[10, 10, 10],  # Small mesh for testing
        t_max=300.0,
        t_step=100.0,
        relax_structure=False,
    )
    
    print("Success:", result.get("success"))
    if result.get("success"):
        print(f"Number of temperatures: {len(result.get('temperatures', []))}")
        kappa = result.get("thermal_conductivity", [])
        temps = result.get("temperatures", [])
        if len(kappa) > 0 and len(temps) > 0:
            for t, k in zip(temps, kappa):
                if k is not None:
                    print(f"  T = {t} K: κ = {k} W/m·K")
    else:
        print("Error:", result.get("error"))
