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
- prepare_calculation
- submit_calculation
- get_calculation_status
- fetch_results
- cancel_calculation
- restart_calculation

Typical workflow:
1. prepare_calculation(engine, structure, calc_type) -> job_id
2. submit_calculation(job_id, resources)
3. get_calculation_status(job_id)   # poll until terminal
4. fetch_results(job_id)
"""

from .lifecycle import (
    cancel_calculation,
    fetch_results,
    get_calculation_status,
    prepare_calculation,
    restart_calculation,
    submit_calculation,
)

__all__ = [
    "prepare_calculation",
    "submit_calculation",
    "get_calculation_status",
    "fetch_results",
    "cancel_calculation",
    "restart_calculation",
]

__version__ = "0.1.0"
