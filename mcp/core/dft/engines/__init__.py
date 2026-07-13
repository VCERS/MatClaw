"""Engine adapter registry.

Add a new DFT/QC code by implementing :class:`~core.dft.engines.base.Engine`
and registering it here — nothing else in the subsystem needs to change.
"""

from __future__ import annotations

from typing import Dict

from .base import Engine, PrepareResult
from .orca import OrcaEngine
from .vasp import VaspEngine

_ENGINES: Dict[str, Engine] = {
    "vasp": VaspEngine(),
    "orca": OrcaEngine(),
}


def get_engine(name: str) -> Engine:
    key = (name or "").lower()
    if key not in _ENGINES:
        raise ValueError(
            f"Unknown engine {name!r}; expected one of {sorted(_ENGINES)}"
        )
    return _ENGINES[key]


def supported_engines() -> list[str]:
    return sorted(_ENGINES)


__all__ = ["Engine", "PrepareResult", "get_engine", "supported_engines"]
