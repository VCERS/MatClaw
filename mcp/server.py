"""
MatClaw MCP Server
"""

from dotenv import load_dotenv
import logging
from mcp.server.fastmcp import FastMCP
from tools.pubchem import (
    pubchem_search_compounds,
    pubchem_get_compound_properties,
    pubchem_get_safety_data,
)
from tools.cod import (
    cod_search_structures,
)
from tools.materials_project import (
    mp_search_materials,
    mp_get_material_properties,
    mp_get_detailed_property_data,
    mp_search_recipe
)
from tools.molport import (
    molport_search_molecules,
    molport_get_molecule_info,
)
from tools.ase import (
    ase_connect_or_create_db,
    ase_store_result,
    ase_query,
    ase_get_atoms,
    ase_list_databases
)
from tools.pymatgen import (
    pymatgen_structure_matcher,
    pymatgen_prototype_builder,
    pymatgen_substitution_generator,
    pymatgen_substitution_predictor,
    pymatgen_ion_exchange_generator,
    pymatgen_perturbation_generator,
    pymatgen_defect_generator,
    pymatgen_disorder_generator,
    pymatgen_structure_editor,
    pymatgen_majority_orderer,
    pymatgen_enumeration_orderer,
    pymatgen_sqs_orderer,
)
from tools.analysis import (
    structure_validator,
    composition_analyzer,
    structure_analyzer,
    structure_fingerprinter,
)
from tools.matgl import (
    matgl_relax_structure,
    matgl_predict_bandgap,
    matgl_predict_eform
)
from tools.matcalc import (
    matcalc_calc_adsorption,
    matcalc_calc_elasticity,
    matcalc_calc_energetics,
    matcalc_calc_eos,
    matcalc_calc_interface,
    matcalc_calc_md,
    matcalc_calc_neb,
    matcalc_calc_phonon,
    matcalc_calc_phonon3,
    matcalc_calc_qha,
    matcalc_calc_surface
)
from tools.chem_llm import (
    predict_molecule_binding,
    predict_molecule_synthesizability,
)
from tools.selection import (
    multi_objective_ranker,
)
from tools.synthesis_planning import (
    synthesis_recipe_quantifier,
)
from tools.elemwise_retro import (
    er_predict_precursors,
    er_predict_temperature,
)
from tools.arrows import (
    arrows_initialize_campaign,
    arrows_suggest_experiment,
    arrows_record_result,
)
from tools.bayesian_optimization import (
    bo_initialize_campaign,
    bo_suggest_experiment,
    bo_record_result,
)
from tools.characterization import (
    xrd_analyze_pattern,
)
from tools.characterization import (
    xrd_analyze_pattern,
)
from tools.urdf import (
    urdf_validate,
    urdf_fix,
    urdf_inspect,
)
from tools.lula import (
    lula_generate_robot_description,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP(name="matclaw-mcp-server")

# Add tools
# COD tools
mcp.tool()(cod_search_structures)
# Pubchem tools
mcp.tool()(pubchem_search_compounds)
mcp.tool()(pubchem_get_compound_properties)
mcp.tool()(pubchem_get_safety_data)

# Materials Project tools
mcp.tool()(mp_search_materials)
mcp.tool()(mp_get_material_properties)
mcp.tool()(mp_get_detailed_property_data)
mcp.tool()(mp_search_recipe)

# Molport tools
mcp.tool()(molport_search_molecules)
mcp.tool()(molport_get_molecule_info)

# ASE database tools
mcp.tool()(ase_connect_or_create_db)
mcp.tool()(ase_store_result)
mcp.tool()(ase_query)
mcp.tool()(ase_get_atoms)
mcp.tool()(ase_list_databases)

# Pymatgen structure generation tools
mcp.tool()(pymatgen_structure_matcher)
mcp.tool()(pymatgen_prototype_builder)
mcp.tool()(pymatgen_substitution_generator)
mcp.tool()(pymatgen_substitution_predictor)
mcp.tool()(pymatgen_ion_exchange_generator)
mcp.tool()(pymatgen_perturbation_generator)
mcp.tool()(pymatgen_defect_generator)
mcp.tool()(pymatgen_disorder_generator)
mcp.tool()(pymatgen_structure_editor)
mcp.tool()(pymatgen_majority_orderer)
mcp.tool()(pymatgen_enumeration_orderer)
mcp.tool()(pymatgen_sqs_orderer)

# Analysis tools for materials screening
mcp.tool()(structure_validator)
mcp.tool()(composition_analyzer)
mcp.tool()(structure_analyzer)
mcp.tool()(structure_fingerprinter)

# Machine learning prediction tools
mcp.tool()(matgl_relax_structure)
mcp.tool()(matgl_predict_bandgap)
mcp.tool()(matgl_predict_eform)

# Material property calculation tools
mcp.tool()(matcalc_calc_adsorption)
mcp.tool()(matcalc_calc_elasticity)
mcp.tool()(matcalc_calc_energetics)
mcp.tool()(matcalc_calc_eos)
mcp.tool()(matcalc_calc_interface)
mcp.tool()(matcalc_calc_md)
mcp.tool()(matcalc_calc_neb)
mcp.tool()(matcalc_calc_phonon)
mcp.tool()(matcalc_calc_phonon3)
mcp.tool()(matcalc_calc_qha)
mcp.tool()(matcalc_calc_surface)

# Fine-tuned LLM prediction tools
mcp.tool()(predict_molecule_binding)
mcp.tool()(predict_molecule_synthesizability)

# Selection and ranking tools
mcp.tool()(multi_objective_ranker)

# Synthesis planning tools
mcp.tool()(synthesis_recipe_quantifier)

# ElemwiseRetro tools
mcp.tool()(er_predict_precursors)
mcp.tool()(er_predict_temperature)

# ARROWS active learning tools
mcp.tool()(arrows_initialize_campaign)
mcp.tool()(arrows_suggest_experiment)
mcp.tool()(arrows_record_result)

# Bayesian optimization tools
mcp.tool()(bo_initialize_campaign)
mcp.tool()(bo_suggest_experiment)
mcp.tool()(bo_record_result)

# XRD analysis tools
mcp.tool()(xrd_analyze_pattern)

# URDF validation and fixing tools
mcp.tool()(urdf_validate)
mcp.tool()(urdf_fix)
mcp.tool()(urdf_inspect)

# Lula robot description generation
mcp.tool()(lula_generate_robot_description)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MatClaw MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport mode (default: stdio)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8500,
        help="Port for streamable-http transport (default: 8500)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for streamable-http transport (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info("Starting MatClaw MCP Server (stdio)")
        mcp.run(transport="stdio")
    else:
        logger.info(
            f"Starting MatClaw MCP Server ({args.transport}) "
            f"on {args.host}:{args.port}"
        )
        # Enable stateless JSON mode for HTTP transport
        mcp.settings.stateless_http = True
        mcp.settings.json_response = True
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        # Disable DNS rebinding protection when binding to a non-local IP
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            mcp.settings.transport_security = None
        mcp.run(transport=args.transport)
