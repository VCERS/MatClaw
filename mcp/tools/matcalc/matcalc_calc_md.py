"""
Molecular Dynamics (MD) simulation tool using matcalc MDCalc.

This tool performs MD simulations on structures using universal ML potentials
to calculate thermodynamic properties, sample phase space, and study dynamics.
"""

from typing import Annotated, Any

import numpy as np
from pydantic import Field
from pymatgen.core import Structure


def matcalc_calc_md(
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
    ensemble: Annotated[
        str,
        Field(
            default="nvt",
            description=(
                "MD ensemble for simulation. Options: 'nvt' (canonical), 'nve' (microcanonical), "
                "'npt' (isothermal-isobaric), 'nvt-nh' (Nose-Hoover), 'npt-nh' (NPT Nose-Hoover), "
                "'langevin', 'nvt-andersen', 'nvt-berendsen', 'npt-berendsen'. Default: 'nvt'."
            )
        )
    ] = "nvt",
    temperature: Annotated[
        float,
        Field(
            default=300.0,
            gt=0.0,
            description="Temperature in Kelvin for the simulation. Default: 300.0."
        )
    ] = 300.0,
    timestep: Annotated[
        float,
        Field(
            default=1.0,
            gt=0.0,
            description=(
                "Time step for MD integration in femtoseconds (fs). "
                "Typical values: 0.5-2.0 fs depending on system dynamics. Default: 1.0."
            )
        )
    ] = 1.0,
    steps: Annotated[
        int,
        Field(
            default=100,
            ge=1,
            description=(
                "Number of MD steps to run. For production runs, use 10000+ steps. "
                "Default: 100 (suitable for testing)."
            )
        )
    ] = 100,
    pressure: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Pressure in GPa for NPT ensemble. Only used if ensemble contains 'npt'. "
                "If None and NPT is used, converts to ~0 GPa internally. Default: None."
            )
        )
    ] = None,
    relax_structure: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether to relax the structure before MD. "
                "Recommended: True to ensure structure is at equilibrium and avoid numerical instabilities. "
                "Default: True."
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
    optimizer: Annotated[
        str,
        Field(
            default="FIRE",
            description=(
                "Optimizer for structure relaxation. "
                "Options: 'FIRE', 'BFGS', 'LBFGS', 'BFGSLineSearch'. Default: 'FIRE'."
            )
        )
    ] = "FIRE",
    trajfile: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Path to save trajectory file (e.g., 'trajectory.traj'). "
                "If None, trajectory is not saved to file. Note: trajectory files can be large. Default: None."
            )
        )
    ] = None,
    logfile: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Path to save MD log file (e.g., 'md.log'). "
                "If None, log is not saved to file. Default: None."
            )
        )
    ] = None,
    loginterval: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description=(
                "Interval (in steps) for logging MD information. "
                "Use values > 1 for long simulations to reduce file size. Default: 1 (log every step)."
            )
        )
    ] = 1,
    taut: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Time constant for Berendsen/Nose-Hoover thermostat in fs. "
                "If None, uses ensemble-specific defaults. Default: None."
            )
        )
    ] = None,
    taup: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Time constant for Berendsen/Nose-Hoover barostat in fs. "
                "If None, uses ensemble-specific defaults. Default: None."
            )
        )
    ] = None,
    friction: Annotated[
        float,
        Field(
            default=0.001,
            gt=0.0,
            description="Friction coefficient for Langevin dynamics in fs^-1. Default: 0.001."
        )
    ] = 0.001,
    relax_calc_kwargs: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Additional keyword arguments for relaxation calculator. Default: None."
        )
    ] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Run molecular dynamics simulation using universal ML potentials in various ensembles (NVE, NVT, NPT, etc.).
    """
    try:
        from matcalc import MDCalc
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

    # Load calculator
    try:
        calc = mtc.load_fp(calculator)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load calculator '{calculator}': {e}",
            "details": "Check that calculator is available using `matgl.get_available_pretrained_models()`"
        }

    # Set up MDCalc
    try:
        md_calc = MDCalc(
            calculator=calc,
            ensemble=ensemble,
            temperature=temperature,
            timestep=timestep,
            steps=steps,
            pressure=pressure,
            relax_structure=relax_structure,
            fmax=fmax,
            optimizer=optimizer,
            trajfile=trajfile,
            logfile=logfile,
            loginterval=loginterval,
            taut=taut,
            taup=taup,
            friction=friction,
            relax_calc_kwargs=relax_calc_kwargs,
            **kwargs,
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize MDCalc: {e}",
        }

    # Run MD simulation
    try:
        result = md_calc.calc(structure)
    except Exception as e:
        return {
            "success": False,
            "error": f"MD simulation failed: {e}",
        }

    # Extract results
    final_energy = result.get("energy", 0.0)
    final_structure = result.get("final_structure", structure)
    
    # Calculate total simulation time in ps
    total_time_ps = (steps * timestep) / 1000.0  # Convert fs to ps

    from pymatgen.io.cif import CifWriter

    # Return formatted result
    return {
        "success": True,
        "energy": float(final_energy),
        "structure": str(CifWriter(final_structure)) if hasattr(final_structure, 'as_dict') else final_structure,
        "relaxed": relax_structure,
        "ensemble": ensemble,
        "temperature": float(temperature),
        "pressure": float(pressure) if pressure is not None else None,
        "steps": int(steps),
        "timestep": float(timestep),
        "total_time": float(total_time_ps),
        "calculator": calculator,
        "units": {
            "energy": "eV",
            "temperature": "K",
            "pressure": "GPa",
            "timestep": "fs",
            "time": "ps",
            "force": "eV/Angstrom"
        }
    }


def _parse_structure(structure_input: str) -> Structure:
    """
    Parse structure from various input formats.
    
    Args:
        structure_input: Structure as string (CIF/POSCAR)
        
    Returns:
        Pymatgen Structure object
    """
    # If already a Structure object, return it
    if isinstance(structure_input, Structure):
        return structure_input
    
    # If string, try to parse as CIF or POSCAR
    if isinstance(structure_input, str):
        # Try CIF first (check for common CIF patterns)
        if structure_input.strip().startswith('data_') or '_cell_' in structure_input:
            try:
                return Structure.from_str(structure_input, fmt='cif')
            except Exception:
                raise ValueError("Failed to parse structure as CIF format")
        
        # Try POSCAR format
        try:
            return Structure.from_str(structure_input, fmt='poscar')
        except Exception:
            raise ValueError("Failed to parse structure as POSCAR format")
    
    raise ValueError(f"Unsupported structure input type: {type(structure_input)}")


def _format_array(arr) -> list[float]:
    """Convert numpy array to list of floats for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, (list, tuple)):
        return [float(x) for x in arr]
    else:
        return [float(arr)]
