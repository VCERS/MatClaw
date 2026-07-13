"""
dft_tools: DFT job-lifecycle MCP tools for VASP and ORCA.

The agent-facing control plane for running periodic (VASP) and molecular (ORCA)
DFT/QC jobs on an HPC without SSH access. Each tool call is short and
non-blocking; long calculations run under the batch scheduler and are tracked by
a persistent ``job_id``.

Design:
- The six lifecycle tools are engine-agnostic; ``engine`` ("vasp"|"orca") is a
  discriminator, not a separate tool family.
- All heavy lifting (config, persistence, scheduler, engine adapters) lives in
  the ``core.dft`` subsystem; these functions only orchestrate it.
- Returns are plain dicts with a ``success`` field, matching the other MatClaw
  tools.

Tools:
- dft_prepare_calculation
- dft_submit_calculation
- dft_get_calculation_status
- dft_fetch_results
- dft_cancel_calculation
- dft_restart_calculation

Typical workflow:
1. dft_prepare_calculation(engine, structure, calc_type) -> job_id
2. dft_submit_calculation(job_id, resources)
3. dft_get_calculation_status(job_id)   # poll until terminal
4. dft_fetch_results(job_id)
"""

from .lifecycle import (
    dft_cancel_calculation,
    dft_fetch_results,
    dft_get_calculation_status,
    dft_prepare_calculation,
    dft_restart_calculation,
    dft_submit_calculation,
)

__all__ = [
    "dft_prepare_calculation",
    "dft_submit_calculation",
    "dft_get_calculation_status",
    "dft_fetch_results",
    "dft_cancel_calculation",
    "dft_restart_calculation",
]

__version__ = "0.1.0"
