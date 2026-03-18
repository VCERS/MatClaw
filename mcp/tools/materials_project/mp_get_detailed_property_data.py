"""
Tool for retrieving detailed property data from Materials Project.
Use this after mp_get_material_properties indicates data is available.

# Visualization
"plot_config" is added as a key in responses which can be visualized as charts, 
which can be intercepted by the backend.
"""

from typing import Dict, Any, Optional, Annotated
from pydantic import Field
from mp_api.client import MPRester
import os
import numpy as np


def mp_get_detailed_property_data(
    material_id: Annotated[
        str, 
        Field(description="Materials Project ID (e.g., 'mp-149')")
    ],
    data_type: Annotated[
        str, 
        Field(description="Type of detailed data to retrieve")
    ],
    element: Annotated[
        Optional[str], 
        Field(default=None, description="Element symbol for XAS spectrum (e.g., 'Si'). Required if data_type='xas_spectrum'")
    ] = None,
    edge: Annotated[
        Optional[str], 
        Field(default="K", description="XAS edge type (K, L1, L2, L3, M). Used only for xas_spectrum")
    ] = "K",
    spectrum_type: Annotated[
        Optional[str], 
        Field(default="XANES", description="XAS spectrum type (XANES, EXAFS, XAFS). Used only for xas_spectrum")
    ] = "XANES",
) -> Dict[str, Any]:
    """
    Get detailed property data for plotting and analysis.
    
    Use this tool AFTER mp_get_material_properties indicates the data is available
    (e.g., band_structure_available=true, dos_available=true, etc.)
    
    Available data types:
    - band_structure: Electronic band structure (k-points, band energies, gaps)
    - dos: Density of states (energy grid, DOS values, optional projections)
    - elastic_tensor: Full elastic tensor (6x6 Cij matrix in GPa)
    - dielectric_tensor: Full dielectric tensor (3x3 matrix)
    - piezoelectric_tensor: Full piezoelectric tensor (3x6 matrix in C/m²)
    - thermal_displacement_data: Temperature-dependent thermal displacement data
    - phonon_bandstructure: Phonon dispersion (q-points, phonon frequencies)
    - phonon_dos: Phonon density of states
    - xas_spectrum: X-ray absorption spectrum (requires element parameter)
    - eos_data: Equation of state volume-energy data points
    
    Args:
        material_id: Materials Project ID
        data_type: Type of data to retrieve
        element: Element for XAS (required if data_type='xas_spectrum')
        edge: XAS edge (K, L1, L2, L3, M)
        spectrum_type: XAS type (XANES, EXAFS, XAFS)
    
    Returns:
        Dictionary with detailed property data or error message
    
    Examples:
        mp_get_detailed_property_data("mp-149", "band_structure")
        mp_get_detailed_property_data("mp-149", "dos")
        mp_get_detailed_property_data("mp-149", "xas_spectrum", element="Si", edge="K")
    """
    try:
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            return {"success": False, "error": "MP_API_KEY environment variable not set"}
        
        with MPRester(api_key) as mpr:
            
            # Route to appropriate handler based on data_type
            if data_type == "band_structure":
                data = _get_band_structure(mpr, material_id)
                return data
            
            elif data_type == "dos":
                data = _get_dos(mpr, material_id)
                return data
            
            elif data_type == "elastic_tensor":
                data = _get_elastic_tensor(mpr, material_id)
                return data
            
            elif data_type == "dielectric_tensor":
                data = _get_dielectric_tensor(mpr, material_id)
                return data
            
            elif data_type == "piezoelectric_tensor":
                data = _get_piezoelectric_tensor(mpr, material_id)
                return data
            
            elif data_type == "thermal_displacement_data":
                data = _get_thermal_displacement_data(mpr, material_id)
                return data

            elif data_type == "phonon_bandstructure":
                data = _get_phonon_bandstructure(mpr, material_id)
                return data
            
            elif data_type == "phonon_dos":
                data = _get_phonon_dos(mpr, material_id)
                return data
            
            elif data_type == "xas_spectrum":
                if not element:
                    return {"success": False, "error": "element parameter required for xas_spectrum"}
                data = _get_xas_spectrum(mpr, material_id, element, edge, spectrum_type)
                return data
            
            elif data_type == "eos_data":
                data = _get_eos_data(mpr, material_id)
                return data
                            
            else:
                return {"success": False, "error": f"Unknown data_type: {data_type}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    

# Utility functions
def _get_band_structure(mpr, material_id: str) -> Dict[str, Any]:
    """Get electronic band structure data"""
    try:
        bs = mpr.get_bandstructure_by_material_id(material_id)
        
        if not bs:
            return {"success": False, "error": f"No band structure data for {material_id}"}
        
        # Extract k-points and branches
        kpoints = []
        labels = []
        branches = []
        
        for branch in bs.branches:
            start_idx, end_idx = branch['start_index'], branch['end_index']
            branches.append({
                "start_index": start_idx,
                "end_index": end_idx,
                "name": branch.get('name', '')
            })
        
        for kpt in bs.kpoints:
            kpoints.append(np.around(kpt.frac_coords, decimals=6).tolist())
            labels.append(kpt.label if kpt.label else "")
        
        # Extract band energies (shift by Fermi energy)
        efermi = bs.efermi if hasattr(bs, 'efermi') else 0.0
        
        bands = {}
        for spin, band_data in bs.bands.items():
            bands[str(spin)] = np.around(band_data - efermi, decimals=4).tolist()
        
        return {
            "success": True,
            "material_id": material_id,
            "kpoints": kpoints,
            "labels": labels,
            "branches": branches,
            "bands": bands,
            "efermi": round(float(efermi), 4),
            "is_spin_polarized": bs.is_spin_polarized,
            "is_metal": bs.is_metal(),
            "direct_gap": bs.get_direct_band_gap() if not bs.is_metal() else None,
            "band_gap": bs.get_band_gap() if not bs.is_metal() else None,
            "plot_config": {
                "plot_type": "line", 
                "title": f"Band Structure - {material_id}",
                "x_axis": {
                    "field": "kpoint_index",
                    "label": "k-point path",
                    "tick_positions": [i for i, lbl in enumerate(labels) if lbl],
                    "tick_labels": [lbl for lbl in labels if lbl]
                },
                "y_axis": {
                    "field": "energy",
                    "label": "Energy (eV)",
                    "unit": "eV"
                },
                "series": [
                    {
                        "name": f"Spin {spin}",
                        "data_path": f"bands.{spin}",
                        "line_width": 2
                    }
                    for spin in bands.keys()
                ],
                "reference_lines": [
                    {
                        "type": "horizontal",
                        "value": round(float(efermi), 4),
                        "label": "Fermi Level",
                    }
                ],
                "annotations": [
                    {
                        "type": "band_gap",
                        "value": bs.get_band_gap()["energy"] if not bs.is_metal() else None,
                        "direct": bs.get_band_gap()["direct"] if not bs.is_metal() else None
                    }
                ] if not bs.is_metal() else []
            }
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_dos(mpr, material_id: str) -> Dict[str, Any]:
    """Get density of states data"""
    try:
        dos = mpr.get_dos_by_material_id(material_id)
        if not dos:
            return {"success": False, "error": f"No DOS data for {material_id}"}
        
        # Total DOS
        energies = np.around(dos.energies, decimals=4).tolist()
        efermi = dos.efermi
        
        total_dos = {}
        for spin, dos_values in dos.densities.items():
            total_dos[str(spin)] = np.around(dos_values, decimals=4).tolist()
        
        result = {
            "success": True,
            "material_id": material_id,
            "energies": energies, # Long list
            "total_dos": total_dos, # long list
            "efermi": round(float(efermi), 4),
            "num_energy_points": len(energies),
            "plot_config": {
                "plot_type": "area",
                "title": f"Density of States - {material_id}",
                "x_axis": {
                    "field": "energies",
                    "label": "Energy (eV)",
                    "unit": "eV"
                },
                "y_axis": {
                    "field": "total_dos",
                    "label": "DOS (states/eV)",
                    "unit": "states/eV"
                },
                "series": [
                    {
                        "name": f"Spin {spin}",
                        "data_path": f"total_dos.{spin}",
                    }
                    for spin in total_dos.keys()
                ],
                "reference_lines": [
                    {
                        "type": "vertical",
                        "value": round(float(efermi), 4),
                        "label": "Fermi Level",
                    }
                ]
            }
        }

        return result
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_elastic_tensor(mpr, material_id: str) -> Dict[str, Any]:
    """Get full elastic tensor"""
    try:
        elasticity = mpr.materials.elasticity.search(material_ids=[material_id])
        
        if not elasticity:
            return {"success": False, "error": f"No elasticity data for {material_id}"}
        
        e = elasticity[0]
        
        elastic_tensor = None
        if hasattr(e, 'elastic_tensor') and e.elastic_tensor:
            if hasattr(e.elastic_tensor, 'ieee_format') and e.elastic_tensor.ieee_format:
                elastic_tensor = _serialize_tuple_tensor(e.elastic_tensor.ieee_format)
            elif hasattr(e.elastic_tensor, 'raw') and e.elastic_tensor.raw is not None:
                elastic_tensor = _serialize_tuple_tensor(e.elastic_tensor.raw)

        compliance_tensor = None
        if hasattr(e, 'compliance_tensor') and e.compliance_tensor:
            if hasattr(e.compliance_tensor, 'ieee_format') and e.compliance_tensor.ieee_format:
                compliance_tensor = _serialize_tuple_tensor(e.compliance_tensor.ieee_format)
            elif hasattr(e.compliance_tensor, 'raw') and e.compliance_tensor.raw is not None:
                compliance_tensor = _serialize_tuple_tensor(e.compliance_tensor.raw)
        
        return {
            "success": True,
            "material_id": material_id,
            "elastic_tensor": elastic_tensor,  # 6x6 in GPa
            "compliance_tensor": compliance_tensor,  # 6x6 in GPa⁻¹
            "units": "GPa",
            "plot_config": {
                "plot_type": "heatmap",
                "title": f"Elastic Tensor (Cij) - {material_id}",
                "data_field": "elastic_tensor",
                "axis_labels": {
                    "x": ["11", "22", "33", "23", "13", "12"],  # Voigt notation
                    "y": ["11", "22", "33", "23", "13", "12"]
                },
                "annotations": {
                    "show_values": True, 
                    "format": ".1f" 
                },
                "layout": {
                    "aspect": "equal", 
                }
            }
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_dielectric_tensor(mpr, material_id: str) -> Dict[str, Any]:
    """Get full dielectric tensor"""
    try:
        dielectric = mpr.materials.dielectric.search(material_ids=[material_id])
        
        if not dielectric:
            return {"success": False, "error": f"No dielectric data for {material_id}"}
        
        d = dielectric[0]
        
        return {
            "success": True,
            "material_id": material_id,
            "total_tensor": _serialize_tuple_tensor(d.total) if hasattr(d, 'total') and d.total is not None else None,
            "ionic_tensor": _serialize_tuple_tensor(d.ionic) if hasattr(d, 'ionic') and d.ionic is not None else None,
            "electronic_tensor": _serialize_tuple_tensor(d.electronic) if hasattr(d, 'electronic') and d.electronic is not None else None,
            "units": "dimensionless"
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_piezoelectric_tensor(mpr, material_id: str) -> Dict[str, Any]:
    """Get full piezoelectric tensor"""
    try:
        piezo = mpr.materials.piezoelectric.search(material_ids=[material_id])
        
        if not piezo:
            return {"success": False, "error": f"No piezoelectric data for {material_id}"}
        
        p = piezo[0]
        
        return {
            "success": True,
            "material_id": material_id,
            "total_tensor": _serialize_tuple_tensor(p.total) if hasattr(p, 'total') and p.total else None,  # 3x6
            "ionic_tensor": _serialize_tuple_tensor(p.ionic) if hasattr(p, 'ionic') and p.ionic else None,  # 3x6
            "electronic_tensor": _serialize_tuple_tensor(p.electronic) if hasattr(p, 'electronic') and p.electronic else None,  # 3x6
            "units": "C/m²"
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_thermal_displacement_data(mpr, material_id: str) -> Dict[str, Any]:
    """Get thermal displacement data"""
    try:
        phonon = mpr.materials.phonon.search(material_ids=[material_id])
        
        if not phonon:
            return {"success": False, "error": f"No phonon data for {material_id}"}
        
        ph = phonon[0]
        
        if not hasattr(ph, 'thermal_displacement_data') or ph.thermal_displacement_data is None:
            return {"success": False, "error": f"No thermal displacement data for {material_id}"}
        
        thermal_data = ph.thermal_displacement_data

        # Extract key thermal properties
        result = {
            "success": True,
            "material_id": material_id
        }
        
        # Debye temperature (most important single metric)
        if hasattr(thermal_data, 'debye_temperature'):
            result["debye_temperature"] = round(float(thermal_data.debye_temperature), 2)
        
        # Temperature-dependent data (if available)
        if hasattr(thermal_data, 'temperatures'):
            result["temperatures"] = thermal_data.temperatures  # K
        
        if hasattr(thermal_data, 'mean_square_displacement'):
            result["mean_square_displacement"] = thermal_data.mean_square_displacement  # Å²
        
        result["units"] = {
            "temperature": "K",
            "debye_temperature": "K",
            "mean_square_displacement": "Å²"
        }
        
        return result
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_phonon_bandstructure(mpr, material_id: str) -> Dict[str, Any]:
    """Get phonon band structure data"""
    try:
        ph_bs = mpr.get_phonon_bandstructure_by_material_id(material_id)
        
        if not ph_bs:
            return {"success": False, "error": f"No phonon band structure for {material_id}"}
        
        # Extract q-points and frequencies
        qpoints = [np.around(qpt, decimals=4).tolist() for qpt in ph_bs.qpoints]
        frequencies = np.around(ph_bs.frequencies, decimals=4).tolist()  # Shape: (num_branches, num_qpoints)
        
        return {
            "success": True,
            "material_id": material_id,
            "qpoints": qpoints,
            "frequencies": frequencies,  # THz
            "units": "THz",
            "plot_config": {
                "plot_type": "line",
                "title": f"Phonon Band Structure - {material_id}",
                "x_axis": {
                    "field": "qpoint_index",
                    "label": "q-point path"
                },
                "y_axis": {
                    "field": "frequencies",
                    "label": "Frequency (THz)",
                    "unit": "THz"
                },
                "series": [
                    {
                        "name": f"Branch {i+1}",
                        "data_path": f"frequencies[{i}]",
                    }
                    for i in range(len(frequencies))
                ],
                "reference_lines": [
                    {
                        "type": "horizontal",
                        "value": 0,
                        "label": "ω = 0",
                    }
                ],
                "annotations": [
                    {
                        "type": "warning",
                        "message": "Contains imaginary modes (negative frequencies)",
                    }
                ]
            }
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_phonon_dos(mpr, material_id: str) -> Dict[str, Any]:
    """Get phonon DOS data"""
    try:
        ph_dos = mpr.get_phonon_dos_by_material_id(material_id)
        
        if not ph_dos:
            return {"success": False, "error": f"No phonon DOS for {material_id}"}
        
        # Extract frequencies and densities
        frequencies = np.around(ph_dos.frequencies, decimals=4).tolist()  # THz
        densities = np.around(ph_dos.densities, decimals=4).tolist()  # states/THz

        return {
            "success": True,
            "material_id": material_id,
            "frequencies": frequencies, # THz
            "densities": densities, # states/THz
            "units": {"frequencies": "THz", "densities": "states/THz"}
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_xas_spectrum(mpr, material_id: str, element: str, edge: str, spectrum_type: str) -> Dict[str, Any]:
    """Get XAS spectrum data"""
    try:
        xas = mpr.materials.xas.search(material_ids=[material_id])
        
        if not xas:
            return {"success": False, "error": f"No XAS data for {material_id}"}
        
        # Find matching spectrum
        matching_spectrum = None
        for spec in xas:
            if (str(spec.absorbing_element) == element and 
                spec.edge.value == edge and 
                spec.spectrum_type.value == spectrum_type):
                matching_spectrum = spec
                break
        
        if not matching_spectrum:
            return {
                "success": False, 
                "error": f"No {spectrum_type} spectrum for {element} {edge}-edge in {material_id}"
            }
        
        # Extract spectrum data
        spectrum_obj = matching_spectrum.spectrum
        
        return {
            "success": True,
            "material_id": material_id,
            "element": element,
            "edge": edge,
            "spectrum_type": spectrum_type,
            "energy": np.around(spectrum_obj.x, decimals=4).tolist(),  # eV
            "intensity": np.around(spectrum_obj.y, decimals=4).tolist(),
            "units": {"energy": "eV", "intensity": "arbitrary"},
            "plot_config": {
                "plot_type": "line",
                "title": f"{element} {edge}-edge {spectrum_type} - {material_id}",
                "x_axis": {
                    "field": "energy",
                    "label": "Energy (eV)",
                    "unit": "eV"
                },
                "y_axis": {
                    "field": "intensity",
                    "label": "Absorption Intensity",
                    "unit": "arbitrary units"
                },
                "series": [
                    {
                        "name": f"{element} {edge}-edge",
                        "x_data_path": "energy",
                        "y_data_path": "intensity",
                    }
                ]
            }
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_eos_data(mpr, material_id: str) -> Dict[str, Any]:
    """Get equation of state volume-energy data"""
    try:
        eos = mpr.materials.eos.search(material_ids=[material_id])
        
        if not eos:
            return {"success": False, "error": f"No EOS data for {material_id}"}
        
        eos_doc = eos[0]
        
        return {
            "success": True,
            "material_id": material_id,
            "volumes": np.around(eos_doc.volumes, decimals=4).tolist() if hasattr(eos_doc, 'volumes') else None,  # Å³
            "energies": np.around(eos_doc.energies, decimals=4).tolist() if hasattr(eos_doc, 'energies') else None,  # eV
            "num_points": len(eos_doc.volumes) if hasattr(eos_doc, 'volumes') else 0,
            "units": {"volume": "Å³", "energy": "eV"},
            "plot_config": {
                "plot_type": "scatter",
                "title": f"Equation of State - {material_id}",
                "x_axis": {
                    "field": "volumes",
                    "label": "Volume (Å³)",
                    "unit": "Å³"
                },
                "y_axis": {
                    "field": "energies",
                    "label": "Energy (eV)",
                    "unit": "eV"
                },
                "series": [
                    {
                        "name": "E-V Data Points",
                        "x_data_path": "volumes",
                        "y_data_path": "energies",
                        "mode": "markers",
                        "marker_size": 8
                    }
                ],
                "annotations": [
                    {
                        "type": "info",
                        "message": f"Total data points: {len(eos_doc.volumes) if hasattr(eos_doc, 'volumes') else 0}"
                    }
                ]
            }
            
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _serialize_tuple_tensor(tensor: tuple[tuple], decimals: int = 4) -> Optional[list[list]]:
    """
    Helper function to convert tuple of tuples to list of lists for JSON serialization.
    Also rounds numeric values to specified decimal places.
    
    Args:
        tensor: Tuple of tuples (or nested structure) to convert
        decimals: Number of decimal places to round to (default: 4)
    
    Returns:
        List of lists with rounded values, or None if input is None
    """
    if tensor is None:
        return None
    try:
        return np.around(np.array(tensor), decimals=decimals).tolist()
    except (ValueError, TypeError):
        return [list(row) for row in tensor]
