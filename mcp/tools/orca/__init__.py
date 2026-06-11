"""
orca_tools: ORCA analysis and cube-generation utilities for MCP workflows.

This package provides structured tools for:
- parsing and summarizing ORCA output files
- validating cube-generation environments
- generating MO, electron-density, and ESP cube files

Package design:
- Public functions return structured dictionaries with a `success` field
- Non-fatal reliability concerns are returned in a `warnings` list
- Analysis tools are read-oriented
- Cube tools depend on `orca_plot`, writable directories, and compatible ORCA behavior

Recommended high-level entry points for skills and MCP:
- orca_summarize_output
- orca_batch_summarize_outputs
- orca_validate_environment
- orca_validate_calc_dir
- orca_generate_homo_lumo_cubes
- orca_generate_density_esp_cubes

Typical workflow:
1. orca_validate_environment()
2. orca_validate_calc_dir()
3. orca_summarize_output() or cube-generation functions
"""

from .orca_analysis_tools import (
    orca_scan_output_files,
    orca_pick_output,
    orca_summarize_output,
    orca_batch_summarize_outputs,
)

from .orca_cube_tools import (
    orca_validate_environment,
    orca_validate_calc_dir,
    orca_find_matching_gbw,
    orca_generate_mo_cube,
    orca_generate_homo_lumo_cubes,
    orca_generate_density_esp_cubes,
)

__all__ = [
    # Analysis tools
    "orca_scan_output_files",
    "orca_pick_output",
    "orca_summarize_output",
    "orca_batch_summarize_outputs",
    # Cube / environment tools
    "orca_validate_environment",
    "orca_validate_calc_dir",
    "orca_find_matching_gbw",
    "orca_generate_mo_cube",
    "orca_generate_homo_lumo_cubes",
    "orca_generate_density_esp_cubes",
]