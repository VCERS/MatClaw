"""
Engine adapter interface — where the physics diverges.

The lifecycle (submit/poll/fetch) is engine-agnostic; only input generation,
the run command, and output parsing differ between VASP (periodic plane-wave)
and ORCA (molecular Gaussian-basis). Each adapter implements this small
interface, and new engines (Quantum ESPRESSO, CP2K, ...) are added by writing
one more adapter — no changes to the tools or the scheduler.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DFTConfig


@dataclass
class PrepareResult:
    """Outcome of staging inputs into a working directory."""

    input_files: Dict[str, str]          # logical name -> absolute path
    resolved_params: Dict[str, Any]      # fully-resolved parameters actually used
    output_file: str                     # primary output file the engine will write
    warnings: List[str] = field(default_factory=list)


class Engine(ABC):
    name: str = "base"
    # Calculation types the adapter understands; the first is the default.
    calc_types: tuple = ()

    @abstractmethod
    def prepare(
        self,
        workdir: str,
        structure: str,
        calc_type: str,
        structure_format: str = "auto",
        charge: int = 0,
        multiplicity: int = 1,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PrepareResult:
        """Write input files into ``workdir`` and report what was resolved."""

    @abstractmethod
    def run_commands(
        self, record_workdir: str, resolved_params: Dict[str, Any],
        resources: Dict[str, Any], config: DFTConfig,
    ) -> List[str]:
        """Return the shell command(s) that execute the calculation."""

    @abstractmethod
    def parse_results(
        self, workdir: str, calc_type: str, resolved_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse completed outputs into a structured, JSON-safe dict."""

    def prepare_restart(
        self,
        parent_workdir: str,
        new_workdir: str,
        resolved_params: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PrepareResult:
        """Default restart: copy every staged input file across verbatim.

        Engines that have a checkpoint (CONTCAR, .gbw) override this to wire it
        in. ``resolved_params`` is the parent job's resolved parameter set.
        """
        src, dst = Path(parent_workdir), Path(new_workdir)
        dst.mkdir(parents=True, exist_ok=True)
        copied: Dict[str, str] = {}
        for name in resolved_params.get("input_files", []):
            f = src / name
            if f.is_file():
                shutil.copy2(f, dst / name)
                copied[name] = str(dst / name)
        return PrepareResult(
            input_files=copied,
            resolved_params=dict(resolved_params),
            output_file=resolved_params.get("output_file", ""),
            warnings=["Restart reused parent inputs verbatim (no checkpoint wired in)."],
        )


# -- shared structure parsing -------------------------------------------------

def looks_like_cif(text: str) -> bool:
    return "data_" in text or "_cell_length" in text or "loop_" in text


def looks_like_xyz(text: str) -> bool:
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return False
    try:
        int(lines[0].strip())
        return True
    except ValueError:
        return False
