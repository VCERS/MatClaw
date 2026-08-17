"""
MatClaw MCP Server
"""

from dotenv import load_dotenv
import logging
import os

from utils.stdio_guard import redirect_stdout_to_stderr

# Route stray stdout writes to stderr so they can't corrupt the MCP stdio JSON-RPC
# stream. Must run before importing tools that may print at import time.
redirect_stdout_to_stderr()

from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP(name="matclaw-mcp-server")
# Disable DNS rebinding protection — server host binding is controlled via --host
mcp.settings.transport_security = None
# Enable stateless JSON mode for HTTP transport (required for multi-worker)
mcp.settings.stateless_http = True
mcp.settings.json_response = True


# ─────────────────────────────────────────────────────────────────────────────
# Tool groups
#
# Each loader imports its own tools and returns them, so a deployment that
# enables only some groups never imports the others. That matters for two
# reasons: an image can ship without the dependencies of the groups it does not
# serve, and every registered tool costs schema tokens in the client's context
# on every turn.
#
# Selection is via the MATCLAW_ENABLED_GROUPS environment variable, not a CLI
# flag, because registration has to happen at import time — with --workers > 1,
# uvicorn imports `server:get_http_app` in each worker process, where the
# __main__ block never runs. Environment is the one source of truth that both
# the parent and the workers see.
# ─────────────────────────────────────────────────────────────────────────────


def _load_cod():
    from tools.cod import cod_search_structures
    return [cod_search_structures]


def _load_pubchem():
    from tools.pubchem import (
        pubchem_search_compounds,
        pubchem_get_compound_properties,
        pubchem_get_safety_data,
    )
    return [
        pubchem_search_compounds,
        pubchem_get_compound_properties,
        pubchem_get_safety_data,
    ]


def _load_materials_project():
    from tools.materials_project import (
        mp_search_materials,
        mp_get_material_properties,
        mp_get_detailed_property_data,
        mp_search_recipe,
    )
    return [
        mp_search_materials,
        mp_get_material_properties,
        mp_get_detailed_property_data,
        mp_search_recipe,
    ]


def _load_molport():
    from tools.molport import (
        molport_search_molecules,
        molport_get_molecule_info,
    )
    return [molport_search_molecules, molport_get_molecule_info]


def _load_ase():
    from tools.ase import (
        ase_connect_or_create_db,
        ase_store_result,
        ase_query,
        ase_get_atoms,
        ase_list_databases,
    )
    return [
        ase_connect_or_create_db,
        ase_store_result,
        ase_query,
        ase_get_atoms,
        ase_list_databases,
    ]


def _load_pymatgen():
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
    return [
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
    ]


def _load_analysis():
    from tools.analysis import (
        structure_validator,
        composition_analyzer,
        structure_analyzer,
        structure_fingerprinter,
    )
    return [
        structure_validator,
        composition_analyzer,
        structure_analyzer,
        structure_fingerprinter,
    ]


def _load_matgl():
    from tools.matgl import (
        matgl_relax_structure,
        matgl_predict_bandgap,
        matgl_predict_eform,
    )
    return [matgl_relax_structure, matgl_predict_bandgap, matgl_predict_eform]


def _load_matcalc():
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
        matcalc_calc_surface,
    )
    return [
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
        matcalc_calc_surface,
    ]


def _load_chem_llm():
    from tools.chem_llm import (
        predict_molecule_binding,
        predict_molecule_synthesizability,
    )
    return [predict_molecule_binding, predict_molecule_synthesizability]


def _load_selection():
    from tools.selection import multi_objective_ranker
    return [multi_objective_ranker]


def _load_synthesis_planning():
    from tools.synthesis_planning import synthesis_recipe_quantifier
    return [synthesis_recipe_quantifier]


def _load_elemwise_retro():
    from tools.elemwise_retro import (
        er_predict_precursors,
        er_predict_temperature,
    )
    return [er_predict_precursors, er_predict_temperature]


def _load_arrows():
    from tools.arrows import (
        arrows_initialize_campaign,
        arrows_suggest_experiment,
        arrows_record_result,
    )
    return [
        arrows_initialize_campaign,
        arrows_suggest_experiment,
        arrows_record_result,
    ]


def _load_bayesian_optimization():
    from tools.bayesian_optimization import (
        bo_initialize_campaign,
        bo_suggest_experiment,
        bo_record_result,
    )
    return [bo_initialize_campaign, bo_suggest_experiment, bo_record_result]


def _load_characterization():
    from tools.characterization import xrd_analyze_pattern
    return [xrd_analyze_pattern]


def _load_urdf():
    from tools.urdf import urdf_validate, urdf_fix, urdf_inspect
    return [urdf_validate, urdf_fix, urdf_inspect]


def _load_lula():
    from tools.lula import lula_generate_robot_description
    return [lula_generate_robot_description]


def _load_dft():
    from tools.dft import (
        dft_prepare_calculation,
        dft_submit_calculation,
        dft_get_calculation_status,
        dft_fetch_results,
        dft_cancel_calculation,
        dft_restart_calculation,
    )
    return [
        dft_prepare_calculation,
        dft_submit_calculation,
        dft_get_calculation_status,
        dft_fetch_results,
        dft_cancel_calculation,
        dft_restart_calculation,
    ]


def _load_orca():
    from tools.orca import (
        orca_scan_output_files,
        orca_pick_output,
        orca_summarize_output,
        orca_batch_summarize_outputs,
        orca_validate_environment,
        orca_validate_calc_dir,
        orca_find_matching_gbw,
        orca_generate_mo_cube,
        orca_generate_homo_lumo_cubes,
        orca_generate_density_esp_cubes,
    )
    return [
        orca_scan_output_files,
        orca_pick_output,
        orca_summarize_output,
        orca_batch_summarize_outputs,
        orca_validate_environment,
        orca_validate_calc_dir,
        orca_find_matching_gbw,
        orca_generate_mo_cube,
        orca_generate_homo_lumo_cubes,
        orca_generate_density_esp_cubes,
    ]


TOOL_GROUPS = {
    "cod": _load_cod,
    "pubchem": _load_pubchem,
    "materials_project": _load_materials_project,
    "molport": _load_molport,
    "ase": _load_ase,
    "pymatgen": _load_pymatgen,
    "analysis": _load_analysis,
    "matgl": _load_matgl,
    "matcalc": _load_matcalc,
    "chem_llm": _load_chem_llm,
    "selection": _load_selection,
    "synthesis_planning": _load_synthesis_planning,
    "elemwise_retro": _load_elemwise_retro,
    "arrows": _load_arrows,
    "bayesian_optimization": _load_bayesian_optimization,
    "characterization": _load_characterization,
    "urdf": _load_urdf,
    "lula": _load_lula,
    "dft": _load_dft,
    "orca": _load_orca,
}


def selected_groups():
    """Resolve which tool groups to register from MATCLAW_ENABLED_GROUPS.

    Unset or "all" means every group, which is the historical behaviour. An
    unknown group name raises rather than being skipped: a typo in a deployment
    config should fail loudly at startup instead of quietly serving a smaller
    toolset that nobody notices for weeks.
    """
    raw = os.getenv("MATCLAW_ENABLED_GROUPS", "").strip()
    if not raw or raw.lower() == "all":
        return list(TOOL_GROUPS)

    names = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = [n for n in names if n not in TOOL_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown tool group(s) in MATCLAW_ENABLED_GROUPS: {', '.join(unknown)}. "
            f"Valid groups: {', '.join(TOOL_GROUPS)}"
        )
    # dict.fromkeys de-duplicates while preserving order
    return list(dict.fromkeys(names))


def register_tools():
    """Register the enabled groups' tools on the server. Returns the tool count.

    A group whose import fails is logged and skipped rather than taking the whole
    server down — one unmet optional dependency should not cost you the other
    nineteen groups.
    """
    total = 0
    skipped = []
    for name in selected_groups():
        try:
            tools = TOOL_GROUPS[name]()
        except Exception as exc:
            logger.error("Tool group '%s' failed to load and was skipped: %s", name, exc)
            skipped.append(name)
            continue
        for tool in tools:
            mcp.tool()(tool)
        total += len(tools)
        logger.info("Registered tool group '%s' (%d tool(s))", name, len(tools))

    if skipped:
        logger.error("Skipped tool group(s): %s", ", ".join(skipped))
    logger.info("MatClaw MCP server exposing %d tool(s)", total)
    return total


# Registration runs at import time so uvicorn workers, which import this module
# rather than execute it, register the same tools as the parent process.
register_tools()


def get_http_app():
    """Return the streamable-http Starlette app (for multi-worker uvicorn)."""
    return mcp.streamable_http_app()


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers for HTTP mode (default: 1)"
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info("Starting MatClaw MCP Server (stdio)")
        mcp.run(transport="stdio")
    else:
        logger.info(
            f"Starting MatClaw MCP Server ({args.transport}) "
            f"on {args.host}:{args.port} with {args.workers} worker(s)"
        )
        mcp.settings.host = args.host
        mcp.settings.port = args.port

        if args.workers > 1:
            import uvicorn
            uvicorn.run(
                "server:get_http_app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level=mcp.settings.log_level.lower(),
                factory=True,
            )
        else:
            mcp.run(transport=args.transport)
