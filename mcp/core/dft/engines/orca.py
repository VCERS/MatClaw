"""
ORCA engine adapter (molecular Gaussian-basis quantum chemistry).

Input generation writes a plain ``orca.inp``; result parsing deliberately
delegates to the project's existing, tested parser
(``tools.orca.orca_summarize_output``) rather than re-implementing it — the new
DFT lifecycle reuses that work instead of duplicating it.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from ..config import DFTConfig
from .base import Engine, PrepareResult, looks_like_cif, looks_like_xyz

# calc_type -> ORCA run keyword(s) appended to the "!" simple-input line.
_KEYWORDS_FOR_CALC = {
    "single_point": "",
    "opt": "Opt",
    "relax": "Opt",
    "freq": "Freq",
    "opt_freq": "Opt Freq",
}

_DEFAULT_METHOD = "B3LYP"
_DEFAULT_BASIS = "def2-SVP"


class OrcaEngine(Engine):
    name = "orca"
    calc_types = ("single_point", "opt", "freq", "opt_freq")

    def _to_xyz_block(self, structure: str, structure_format: str) -> tuple[str, List[str]]:
        """Return cartesian ``element x y z`` lines plus any warnings."""
        from pymatgen.core import Molecule, Structure

        warnings: List[str] = []
        fmt = structure_format
        if fmt == "auto":
            if looks_like_xyz(structure):
                fmt = "xyz"
            elif looks_like_cif(structure):
                fmt = "cif"
            else:
                fmt = "poscar"

        if fmt == "xyz":
            mol = Molecule.from_str(structure, fmt="xyz")
            sites = [(s.specie.symbol, s.coords) for s in mol]
        else:
            # Periodic input given to a molecular engine: take the atoms as a
            # finite cluster and warn — ORCA has no lattice.
            struct = Structure.from_str(structure, fmt=fmt)
            warnings.append(
                "A periodic structure was given to ORCA (a molecular engine); "
                "using its atoms as a finite cluster with no periodicity."
            )
            sites = [(s.specie.symbol, s.coords) for s in struct]

        lines = [f"{el}  {xyz[0]:.8f}  {xyz[1]:.8f}  {xyz[2]:.8f}" for el, xyz in sites]
        return "\n".join(lines), warnings

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
        overrides = overrides or {}
        warnings: List[str] = []
        if calc_type not in self.calc_types:
            warnings.append(
                f"calc_type '{calc_type}' is not a runnable ORCA template "
                f"(supported: {', '.join(self.calc_types)}). Falling back to "
                "'single_point'. Capabilities like TD-DFT, NEB-TS, scans, or "
                "multireference must be expressed via `overrides` (keywords/blocks) "
                "or by extending this engine adapter — do NOT assume the intended "
                "calculation ran."
            )
            calc_type = "single_point"

        method = overrides.get("method", _DEFAULT_METHOD)
        basis = overrides.get("basis", _DEFAULT_BASIS)
        run_kw = _KEYWORDS_FOR_CALC.get(calc_type, "")
        extra = overrides.get("keywords", "")  # e.g. "TightSCF D3BJ"

        xyz_block, xyz_warn = self._to_xyz_block(structure, structure_format)
        warnings.extend(xyz_warn)

        simple_input = " ".join(p for p in ["!", method, basis, run_kw, extra] if p).strip()
        nprocs = overrides.get("nprocs")
        blocks = []
        if nprocs:
            blocks.append(f"%pal nprocs {nprocs} end")
        blocks.extend(overrides.get("blocks", []))  # arbitrary %-blocks

        content = "\n".join(
            [simple_input, *blocks, f"* xyz {charge} {multiplicity}", xyz_block, "*", ""]
        )

        Path(workdir).mkdir(parents=True, exist_ok=True)
        inp_path = Path(workdir) / "orca.inp"
        inp_path.write_text(content)

        resolved = {
            "calc_type": calc_type,
            "method": method,
            "basis": basis,
            "charge": charge,
            "multiplicity": multiplicity,
            "run_keywords": run_kw,
            "input_files": ["orca.inp"],
            "output_file": "orca.out",
        }
        return PrepareResult(
            input_files={"orca.inp": str(inp_path)},
            resolved_params=resolved,
            output_file="orca.out",
            warnings=warnings,
        )

    def run_commands(
        self, record_workdir: str, resolved_params: Dict[str, Any],
        resources: Dict[str, Any], config: DFTConfig,
    ) -> List[str]:
        orca_bin = config.engines.orca_bin or ""
        cmd = Template(config.engines.orca_command).safe_substitute(
            orca_bin=orca_bin, input_file="orca.inp", output_file="orca.out"
        )
        # ORCA requires its full path for parallel runs; emit a hint if unset.
        return [cmd]

    def parse_results(
        self, workdir: str, calc_type: str, resolved_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        out_path = Path(workdir) / "orca.out"
        if not out_path.is_file():
            return {"parsed": False, "error": "orca.out not found; the calculation may not have run."}

        # Reuse the existing, tested ORCA summariser.
        try:
            from tools.orca import orca_summarize_output
        except Exception as exc:  # pragma: no cover - import-path safety net
            return {
                "parsed": False,
                "error": f"Could not import tools.orca.orca_summarize_output: {exc}",
            }

        try:
            summary = orca_summarize_output(str(out_path))
        except Exception as exc:
            return {"parsed": False, "error": f"orca_summarize_output raised: {exc}"}
        return {"parsed": True, "summary": summary}

    def prepare_restart(
        self,
        parent_workdir: str,
        new_workdir: str,
        resolved_params: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PrepareResult:
        import shutil

        src, dst = Path(parent_workdir), Path(new_workdir)
        dst.mkdir(parents=True, exist_ok=True)
        warnings: List[str] = []
        copied: Dict[str, str] = {}

        for name in ("orca.inp",):
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
                copied[name] = str(dst / name)
        gbw = src / "orca.gbw"
        if gbw.is_file():
            shutil.copy2(gbw, dst / "orca.gbw")
            copied["orca.gbw"] = str(dst / "orca.gbw")
            warnings.append(
                "Copied orca.gbw for an initial-guess restart; add '! MOREAD' and "
                "'%moinp \"orca.gbw\"' to the input to actually read it."
            )
        else:
            warnings.append("No orca.gbw checkpoint found; restart begins from scratch.")

        resolved = dict(resolved_params)
        resolved["input_files"] = list(copied.keys())
        return PrepareResult(
            input_files=copied,
            resolved_params=resolved,
            output_file=resolved.get("output_file", "orca.out"),
            warnings=warnings,
        )
