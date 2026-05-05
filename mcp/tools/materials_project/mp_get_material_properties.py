"""
Tool for retrieving detailed properties of specific materials from Materials Project.
Requires MP_API_KEY environment variable with your Materials Project API key.
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import Field
from mp_api.client import MPRester
import numpy as np
import os


def mp_get_material_properties(
    material_ids: Annotated[
        str | List[str], 
        Field(description="Materials Project ID(s) (e.g., 'mp-149' for Silicon or ['mp-149', 'mp-19017']). Can be a single ID or list of IDs.")
    ],
    properties: Annotated[
        Optional[List[str]], 
        Field(
            default=None, 
            description="List of property categories to retrieve. If None, returns comprehensive default set. "
            "Available categories: 'basic', 'structure', 'electronic', 'thermodynamic', 'mechanical', "
            "'magnetic', 'dielectric', 'piezoelectric', 'surface', 'grain_boundary', 'phonon', 'eos', 'xas', 'all'. "
            "Use 'all' for complete property set."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Get detailed properties for specific Materials Project materials by material ID.
    
    Use this AFTER mp_search_materials to get in-depth property information including
    crystal structure details, electronic properties, mechanical properties,
    thermodynamic data, and more.
    
    Property Categories:
        - basic: Formula, elements, composition, space group, density, volume, etc.
        - structure: Crystal structure as CIF string (standard format), plus symmetry metadata
        - electronic: Band gap, band structure, DOS, electronic structure
        - thermodynamic: Formation energy, energy above hull, stability
        - mechanical: Elastic tensor, bulk modulus, shear modulus, Poisson ratio
        - magnetic: Magnetic ordering, magnetic moments
        - dielectric: Dielectric tensor, refractive index
        - piezoelectric: Piezoelectric tensor
        - surface: Surface energy data
        - phonon: Phonon band structure, DOS
        - eos: Equation of state
        - xas: X-ray absorption spectra
    
    Examples:
        - Default properties: material_ids="mp-149"
        - Specific categories: material_ids="mp-149", properties=["electronic", "mechanical"]
        - Get structure as CIF: material_ids="mp-149", properties=["structure"]
        - All properties: material_ids="mp-149", properties=["all"]
        - Multiple materials: material_ids=["mp-149", "mp-19017"], properties=["basic", "electronic"]
    
    Args:
        material_ids: Single material ID (str) or list of IDs (e.g., "mp-149" or ["mp-149", "mp-19017"])
        properties: List of property categories to retrieve. None = default comprehensive set.
    
    Returns:
        Dictionary containing:
            - success: Boolean indicating if retrieval succeeded
            - count: Number of materials with properties retrieved
            - properties: List of property dictionaries (one per material)
                - structure.cif: CIF string (when 'structure' requested)
                - structure.space_group_symbol, crystal_system, etc.: Metadata
            - warnings: List of any errors for individual materials
            - error: Error message if complete failure
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            error_msg = "MP_API_KEY environment variable not set. Get your API key from https://materialsproject.org/api"
            return {
                "success": False,
                "count": 0,
                "properties": [],
                "error": error_msg
            }
        
        # Normalize material_ids to list
        if isinstance(material_ids, str):
            mpid_list = [material_ids]
        else:
            mpid_list = list(material_ids)
        
        # Define default property categories if none specified
        if properties is None:
            properties = ["basic", "structure", "electronic", "thermodynamic", "mechanical", "magnetic"]
        elif "all" in properties:
            properties = ["basic", "structure", "electronic", "thermodynamic", "mechanical", 
                        "magnetic", "dielectric", "piezoelectric", "surface", "phonon", "eos", "xas"]
        
        # Initialize Materials Project API client
        with MPRester(api_key) as mpr:
            
            all_properties = []
            errors = []
            
            # Get properties for each material
            for mpid in mpid_list:
                try:
                    material_props = {
                        "material_id": mpid
                    }

                    # Basic Properties
                    if "basic" in properties:
                        try:
                            summary = mpr.materials.summary.get_data_by_id(mpid)
                            
                            material_props["basic"] = {
                                "formula": summary.formula_pretty,
                                "formula_reduced": str(summary.composition_reduced),
                                "formula_anonymous": summary.formula_anonymous,
                                "elements": [el.value for el in summary.elements],
                                "nelements": summary.nelements,
                                "nsites": summary.nsites,
                                "composition": {str(k): v for k, v in summary.composition.items()},
                                "density": round(summary.density, 4) if summary.density else None,
                                "density_atomic": round(summary.density_atomic, 4) if summary.density_atomic else None,
                                "volume": round(summary.volume, 4) if summary.volume else None,
                                "theoretical": summary.theoretical,
                                "database_ids": summary.database_IDs if hasattr(summary, 'database_IDs') else {}
                            }

                        except Exception as e:
                            errors.append(f"Could not fetch basic/structure data for {mpid}")
                    
                    # Structure Properties
                    if "structure" in properties:
                        try:
                            summary = mpr.materials.summary.get_data_by_id(mpid)
                            
                            # Return structure as CIF string (standard crystallographic format)
                            # Includes symmetry metadata in the response for reference
                            material_props["structure"] = {
                                "cif": summary.structure.to(fmt="cif") if summary.structure else None,
                                "crystal_system": summary.symmetry.crystal_system.value if summary.symmetry else None,
                                "space_group_symbol": summary.symmetry.symbol if summary.symmetry else None,
                                "space_group_number": summary.symmetry.number if summary.symmetry else None,
                                "point_group": summary.symmetry.point_group if summary.symmetry else None,
                                "num_sites": len(summary.structure) if summary.structure else 0
                            }
                        
                        except Exception as e:
                            errors.append(f"Could not fetch basic/structure data for {mpid}")
                    
                    # Electronic Properties
                    if "electronic" in properties:
                        try:
                            summary = mpr.materials.summary.get_data_by_id(mpid)
                            
                            material_props["electronic"] = {
                                "band_gap": round(summary.band_gap, 4) if summary.band_gap is not None else None,
                                "is_gap_direct": summary.is_gap_direct,
                                "is_metal": summary.is_metal,
                                "cbm": round(summary.cbm, 4) if hasattr(summary, 'cbm') and summary.cbm else None,
                                "vbm": round(summary.vbm, 4) if hasattr(summary, 'vbm') and summary.vbm else None,
                                "efermi": round(summary.efermi, 4) if hasattr(summary, 'efermi') and summary.efermi else None
                            }
                            
                            # Try to get electronic structure details
                            try:
                                electronic_structure = mpr.materials.electronic_structure.search(material_ids=[mpid])
                                if electronic_structure:
                                    es = electronic_structure[0]

                                    # DOS availability
                                    if hasattr(es, 'dos') and es.dos is not None:
                                        material_props["electronic"]["dos_available"] = True # NOTE: Large data, can fetch via another tool call
                                    
                                    # Band structure availability + key info
                                    if hasattr(es, 'bandstructure') and es.bandstructure is not None:
                                        bs = es.bandstructure
                                        material_props["electronic"]["band_structure_available"] = True # NOTE: Large data, can fetch via another tool call
                                        material_props["electronic"]["is_spin_polarized"] = bs.is_spin_polarized if hasattr(bs, 'is_spin_polarized') else False
                                        
                                        # k-point info for direct/indirect gap analysis
                                        try:
                                            cbm_info = bs.get_cbm()
                                            vbm_info = bs.get_vbm()
                                            material_props["electronic"]["cbm_kpoint"] = cbm_info['kpoint'].frac_coords.tolist() if cbm_info else None
                                            material_props["electronic"]["vbm_kpoint"] = vbm_info['kpoint'].frac_coords.tolist() if vbm_info else None
                                        except:
                                            pass
                            except:
                                pass
                        
                        except Exception as e:
                            errors.append(f"Could not fetch electronic data for {mpid}")
                    
                    # Thermodynamic Properties
                    if "thermodynamic" in properties:
                        try:
                            thermo = mpr.materials.thermo.search(material_ids=[mpid])
                            
                            if thermo:
                                # Preference order GGA_GGA+U_R2SCAN > r2SCAN > GGA_GGA+U
                                preferred_functionals = ["GGA_GGA+U_R2SCAN", "R2SCAN", "GGA_GGA+U"]
                                
                                thermo_doc = None
                                for functional in preferred_functionals:
                                    for t in thermo:
                                        if t.thermo_type.value == functional:
                                            thermo_doc = t
                                            break
                                    if thermo_doc:
                                        break
                                
                                # Fallback to first if no match (shouldn't happen)
                                if not thermo_doc:
                                    thermo_doc = thermo[0]
                                
                                material_props["thermodynamic"] = {
                                    "formation_energy_per_atom": round(thermo_doc.formation_energy_per_atom, 4) if thermo_doc.formation_energy_per_atom else None,
                                    "energy_above_hull": round(thermo_doc.energy_above_hull, 4) if thermo_doc.energy_above_hull is not None else None,
                                    "is_stable": thermo_doc.is_stable,
                                    "equilibrium_reaction_energy_per_atom": round(thermo_doc.equilibrium_reaction_energy_per_atom, 4) if hasattr(thermo_doc, 'equilibrium_reaction_energy_per_atom') and thermo_doc.equilibrium_reaction_energy_per_atom else None,
                                    "decomposes_to": [str(p.material_id) for p in thermo_doc.decomposes_to] if hasattr(thermo_doc, 'decomposes_to') and thermo_doc.decomposes_to else None,
                                    "uncorrected_energy_per_atom": round(thermo_doc.uncorrected_energy_per_atom, 4) if hasattr(thermo_doc, 'uncorrected_energy_per_atom') and thermo_doc.uncorrected_energy_per_atom else None,
                                    "functional": thermo_doc.thermo_type.value,  # Track which functional was used
                                }
                            else:
                                material_props["thermodynamic"] = {"error": "No thermodynamic data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch thermodynamic data for {mpid}")
                    
                    # Mechanical Properties (Elasticity)
                    if "mechanical" in properties:
                        try:
                            elasticity = mpr.materials.elasticity.search(material_ids=[mpid])
                            
                            if elasticity:
                                e = elasticity[0]
                                material_props["mechanical"] = {
                                    "bulk_modulus_vrh": round(e.bulk_modulus.vrh, 2) if e.bulk_modulus.vrh else None,  # GPa
                                    "shear_modulus_vrh": round(e.shear_modulus.vrh, 2) if e.shear_modulus.vrh else None,  # GPa
                                    "elastic_anisotropy": round(e.universal_anisotropy, 4) if e.universal_anisotropy else None,
                                    "poisson_ratio": round(e.homogeneous_poisson, 4) if e.homogeneous_poisson else None,
                                    "bulk_modulus_voigt": round(e.bulk_modulus.voigt, 2) if e.bulk_modulus.voigt else None,
                                    "bulk_modulus_reuss": round(e.bulk_modulus.reuss, 2) if e.bulk_modulus.reuss else None,
                                    "shear_modulus_voigt": round(e.shear_modulus.voigt, 2) if e.shear_modulus.voigt else None,
                                    "shear_modulus_reuss": round(e.shear_modulus.reuss, 2) if e.shear_modulus.reuss else None,
                                    "elastic_tensor_available": hasattr(e, 'elastic_tensor') and e.elastic_tensor is not None, # NOTE: Large data, can fetch via another tool call
                                }
                            else:
                                material_props["mechanical"] = {"info": "No mechanical property data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch mechanical data for {mpid}")
                    
                    # Magnetic Properties
                    if "magnetic" in properties:
                        try:
                            summary = mpr.materials.summary.get_data_by_id(mpid)
                            
                            material_props["magnetic"] = {
                                "is_magnetic": summary.is_magnetic,
                                "total_magnetization": round(summary.total_magnetization, 4) if summary.total_magnetization else 0.0,
                                "total_magnetization_normalized_vol": round(summary.total_magnetization_normalized_vol, 6) if hasattr(summary, 'total_magnetization_normalized_vol') and summary.total_magnetization_normalized_vol else None,
                                "total_magnetization_normalized_formula_units": round(summary.total_magnetization_normalized_formula_units, 6) if hasattr(summary, 'total_magnetization_normalized_formula_units') and summary.total_magnetization_normalized_formula_units else None,
                                "num_magnetic_sites": summary.num_magnetic_sites if hasattr(summary, 'num_magnetic_sites') else None,
                                "num_unique_magnetic_sites": summary.num_unique_magnetic_sites if hasattr(summary, 'num_unique_magnetic_sites') else None
                            }
                        
                        except Exception as e:
                            errors.append(f"Could not fetch magnetic data for {mpid}")
                    
                    # Dielectric Properties
                    if "dielectric" in properties:
                        try:
                            dielectric = mpr.materials.dielectric.search(material_ids=[mpid])
                            
                            if dielectric:
                                d = dielectric[0]
                                material_props["dielectric"] = {
                                    "e_total": round(d.e_total, 4) if d.e_total else None,
                                    "e_ionic": round(d.e_ionic, 4) if d.e_ionic else None,
                                    "e_electronic": round(d.e_electronic, 4) if d.e_electronic else None,
                                    "refractive_index": round(d.n, 4) if d.n else None,
                                    "dielectric_tensor_available": hasattr(d, 'e_ionic_tensor') or hasattr(d, 'e_electronic_tensor') # NOTE: large data, can fetch via another tool call
                                }
                            else:
                                material_props["dielectric"] = {"info": "No dielectric data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch dielectric data for {mpid}")
                    
                    # Piezoelectric Properties
                    if "piezoelectric" in properties:
                        try:
                            piezo = mpr.materials.piezoelectric.search(material_ids=[mpid])
                            
                            if piezo:
                                p = piezo[0]
                                material_props["piezoelectric"] = {
                                    "e_ij_max": round(p.e_ij_max, 4) if p.e_ij_max else None,  # C/m² - Maximum piezoelectric coefficient
                                    "max_direction": p.max_direction if hasattr(p, 'max_direction') and p.max_direction else None,  # [x, y, z] - Direction of maximum response
                                    "strain_for_max": p.strain_for_max if hasattr(p, 'strain_for_max') and p.strain_for_max else None,  # Voigt notation - Strain that produces max
                                    "piezoelectric_tensor_available": hasattr(p, 'total') and p.total is not None,  # NOTE: Large data, can fetch via another tool call
                                }
                                # Calculate ionic/electronic contributions
                                if hasattr(p, 'ionic') and p.ionic is not None:
                                    ionic_array = np.array(p.ionic)
                                    material_props["piezoelectric"]["ionic_max"] = round(float(np.max(np.abs(ionic_array))), 4)
                                if hasattr(p, 'electronic') and p.electronic is not None:
                                    electronic_array = np.array(p.electronic)
                                    material_props["piezoelectric"]["electronic_max"] = round(float(np.max(np.abs(electronic_array))), 4)
                            else:
                                material_props["piezoelectric"] = {"info": "No piezoelectric data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch piezoelectric data for {mpid}")
                    
                    # Surface Properties
                    if "surface" in properties:
                        try:
                            surface = mpr.materials.surface_properties.search(material_ids=[mpid])
                            
                            if surface:
                                s = surface[0]
                                material_props["surface"] = {
                                    "weighted_surface_energy": round(s.weighted_surface_energy, 4) if s.weighted_surface_energy else None,
                                    "weighted_work_function": round(s.weighted_work_function, 4) if s.weighted_work_function else None,
                                    "surface_anisotropy": round(s.surface_anisotropy, 4) if s.surface_anisotropy else None,
                                    "shape_factor": round(s.shape_factor, 4) if s.shape_factor else None,
                                    "num_facets_calculated": len(s.surfaces),
                                    "most_stable_facet": {
                                        "miller_index": sorted(s.surfaces, key=lambda x: x.surface_energy)[0].miller_index,
                                        "surface_energy": round(sorted(s.surfaces, key=lambda x: x.surface_energy)[0].surface_energy, 4)
                                    },
                                    "largest_wulff_facet": {
                                        "miller_index": max([f for f in s.surfaces if f.has_wulff], key=lambda x: x.area_fraction).miller_index,
                                        "area_fraction": round(max([f for f in s.surfaces if f.has_wulff], key=lambda x: x.area_fraction).area_fraction, 4)
                                    } if any(f.has_wulff for f in s.surfaces) else None,
                                }
                            else:
                                material_props["surface"] = {"info": "No surface property data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch surface data for {mpid}")
                    
                    # Phonon Properties
                    if "phonon" in properties:
                        try:
                            phonon = mpr.materials.phonon.search(material_ids=[mpid])
                            
                            if phonon:
                                ph = phonon[0]
                                
                                material_props["phonon"] = {
                                    "thermal_displacement_data_available": hasattr(ph, 'thermal_displacement_data') and ph.thermal_displacement_data is not None, # NOTE: Large data, can fetch via another tool call
                                    "phonon_bandstructure_available": hasattr(ph, 'ph_bs') and ph.ph_bs is not None, # NOTE: Large data, can fetch via another tool call
                                    "phonon_dos_available": hasattr(ph, 'ph_dos') and ph.ph_dos is not None # NOTE: Large data, can fetch via another tool call
                                }
                                
                                # Calculate epsilon averages (for isotropic/cubic materials)
                                if hasattr(ph, 'epsilon_static') and ph.epsilon_static is not None:
                                    epsilon_static_tensor = np.array(ph.epsilon_static)
                                    # Average of diagonal elements
                                    epsilon_static_avg = np.mean([
                                        epsilon_static_tensor[0, 0], 
                                        epsilon_static_tensor[1, 1], 
                                        epsilon_static_tensor[2, 2]
                                    ])
                                    material_props["phonon"]["epsilon_static_average"] = round(float(epsilon_static_avg), 4)
                                
                                if hasattr(ph, 'epsilon_electronic') and ph.epsilon_electronic is not None:
                                    epsilon_electronic_tensor = np.array(ph.epsilon_electronic)
                                    epsilon_electronic_avg = np.mean([
                                        epsilon_electronic_tensor[0, 0],
                                        epsilon_electronic_tensor[1, 1],
                                        epsilon_electronic_tensor[2, 2]
                                    ])
                                    material_props["phonon"]["epsilon_electronic_average"] = round(float(epsilon_electronic_avg), 4)
                                
                                # Calculate ionic contribution (static - electronic)
                                if material_props["phonon"]["epsilon_static_average"] and material_props["phonon"]["epsilon_electronic_average"]:
                                    ionic_contrib = material_props["phonon"]["epsilon_static_average"] - material_props["phonon"]["epsilon_electronic_average"]
                                    material_props["phonon"]["ionic_contribution_to_epsilon"] = round(ionic_contrib, 4)
                                
                                # Extract max Born effective charge (indicates ionic character)
                                if hasattr(ph, 'born') and ph.born is not None:
                                    born_charges = []
                                    for atom_born in ph.born:
                                        # Each atom has 3x3 Born effective charge tensor
                                        born_tensor = np.array(atom_born)
                                        # Take absolute value of diagonal elements
                                        born_charges.extend([
                                            abs(born_tensor[0, 0]), 
                                            abs(born_tensor[1, 1]), 
                                            abs(born_tensor[2, 2])
                                        ])
                                    
                                    if born_charges:
                                        material_props["phonon"]["max_born_charge"] = round(float(max(born_charges)), 4)
        
                                # Sum rule breaking (quality indicators)
                                if hasattr(ph, 'sum_rules_breaking') and ph.sum_rules_breaking is not None:
                                    sr = ph.sum_rules_breaking
                                    material_props["phonon"]["acoustic_sum_rule_breaking"] = round(float(sr.asr), 6) if hasattr(sr, 'asr') and sr.asr is not None else None
                                    material_props["phonon"]["charge_neutrality_sum_rule_breaking"] = round(float(sr.cnsr), 6) if hasattr(sr, 'cnsr') and sr.cnsr is not None else None
                            
                            else:
                                material_props["phonon"] = {"info": "No phonon data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch phonon data for {mpid}")
                    
                    # Equation of State
                    if "eos" in properties:
                        try:
                            eos = mpr.materials.eos.search(material_ids=[mpid])
                            
                            if eos:
                                eos_doc = eos[0]
                                
                                material_props["eos"] = {
                                    "recommended_model": None,
                                    "equilibrium_volume": None,  # Å³
                                    "bulk_modulus": None,  # GPa
                                    "bulk_modulus_derivative": None,  # Dimensionless (B')
                                    "ground_state_energy": None,  # eV
                                    "num_volume_energy_points": len(eos_doc.volumes) if hasattr(eos_doc, 'volumes') and eos_doc.volumes else 0,
                                    "num_eos_fits": len(eos_doc.eos) if hasattr(eos_doc, 'eos') and eos_doc.eos else 0,
                                    "available_eos_models": [],
                                    "full_eos_data_available": (
                                        hasattr(eos_doc, 'volumes') and eos_doc.volumes is not None and
                                        hasattr(eos_doc, 'energies') and eos_doc.energies is not None
                                    ) # NOTE: Large data, can fetch via another tool call
                                }
                                
                                # Extract EOS fits
                                if hasattr(eos_doc, 'eos') and eos_doc.eos:
                                    # Preferred EOS models (in order of reliability for DFT data)
                                    preferred_models = ['birch_murnaghan', 'vinet', 'murnaghan']
                                    
                                    # List all available models
                                    material_props["eos"]["available_eos_models"] = [
                                        fit.model.value for fit in eos_doc.eos
                                    ]
                                    
                                    selected_fit = None
                                    for pref_model in preferred_models:
                                        for fit in eos_doc.eos:
                                            if fit.model.value == pref_model:
                                                selected_fit = fit
                                                break
                                        if selected_fit:
                                            break
                                    
                                    # Fallback to first available if no preferred found
                                    if not selected_fit:
                                        selected_fit = eos_doc.eos[0]
                                    
                                    # Extract parameters from selected fit
                                    if selected_fit:
                                        material_props["eos"]["recommended_model"] = selected_fit.model.value
                                        material_props["eos"]["equilibrium_volume"] = round(float(selected_fit.V0), 4) if hasattr(selected_fit, 'V0') and selected_fit.V0 else None
                                        material_props["eos"]["bulk_modulus"] = round(float(selected_fit.B0), 2) if hasattr(selected_fit, 'B0') and selected_fit.B0 else None
                                        material_props["eos"]["bulk_modulus_derivative"] = round(float(selected_fit.B1), 4) if hasattr(selected_fit, 'B1') and selected_fit.B1 else None
                                        material_props["eos"]["ground_state_energy"] = round(float(selected_fit.E0), 6) if hasattr(selected_fit, 'E0') and selected_fit.E0 else None

                            else:
                                material_props["eos"] = {"info": "No equation of state data available"}
                        
                        except Exception as e:
                            errors.append(f"Could not fetch EOS data for {mpid}")
                    
                    # XAS (X-ray Absorption Spectra)
                    if "xas" in properties:
                        try:
                            xas = mpr.materials.xas.search(material_ids=[mpid])
                            
                            if xas:
                                # Basic counts
                                material_props["xas"] = {
                                    "total_spectra": len(xas),
                                    "absorbing_elements": sorted(list(set([str(x.absorbing_element) for x in xas]))),
                                    "num_absorbing_elements": len(set([str(x.absorbing_element) for x in xas])),
                                    "spectra_by_element": {},
                                    "xas_spectrum_available": any(x.spectrum for x in xas)  # NOTE: Large data, can fetch via another tool call
                                }

                                # Organize by absorbing element
                                element_spectra = {}
                                for spectrum in xas:
                                    elem = str(spectrum.absorbing_element)
                                    
                                    # Initialize element entry
                                    if elem not in element_spectra:
                                        element_spectra[elem] = {
                                            "edges": [],
                                            "spectrum_types": [],
                                            "spectra_details": []
                                        }
                                    
                                    # Collect unique edges
                                    edge_str = spectrum.edge.value
                                    if edge_str not in element_spectra[elem]["edges"]:
                                        element_spectra[elem]["edges"].append(edge_str)
                                    
                                    # Collect unique spectrum types
                                    spec_type = spectrum.spectrum_type.value
                                    if spec_type not in element_spectra[elem]["spectrum_types"]:
                                        element_spectra[elem]["spectrum_types"].append(spec_type)
                                    
                                    # Add detailed entry
                                    element_spectra[elem]["spectra_details"].append({
                                        "edge": edge_str,
                                        "type": spec_type,
                                        "spectrum_id": str(spectrum.spectrum_id) if hasattr(spectrum, 'spectrum_id') else None
                                    })
                                
                                # Sort for consistent output
                                for elem in element_spectra:
                                    element_spectra[elem]["edges"].sort()
                                    element_spectra[elem]["spectrum_types"].sort()
                                
                                material_props["xas"]["spectra_by_element"] = element_spectra
                                
                            else:
                                material_props["xas"] = {"info": "No XAS data available"}
                        except Exception as e:
                            errors.append(f"Could not fetch XAS data for {mpid}")
                    
                    all_properties.append(material_props)
                
                except Exception as e:
                    error_msg = f"Error getting properties for material {mpid}: {str(e)}"
                    errors.append(error_msg)
                    continue
            
            # Prepare response
            response = {
                "success": len(all_properties) > 0,
                "count": len(all_properties),
                "properties": all_properties,
                "requested_material_ids": mpid_list,
                "requested_property_categories": properties
            }
            
            # Add warnings if some materials failed
            if errors:
                response["warnings"] = errors
            
            # Add error if complete failure
            if len(all_properties) == 0:
                response["error"] = "No properties retrieved for any material"
            
            return response
    
    except Exception as e:
        error_msg = f"Error retrieving material properties: {str(e)}"
        return {
            "success": False,
            "count": 0,
            "properties": [],
            "requested_material_ids": mpid_list if isinstance(material_ids, list) else [material_ids],
            "requested_property_categories": properties,
            "error": error_msg
        }
