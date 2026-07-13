"""
VASP engine adapter (periodic plane-wave DFT) built on pymatgen input sets.

Input generation uses pymatgen's curated ``MPRelaxSet`` / ``MPStaticSet`` so the
defaults are sane and reproducible; ``overrides`` lets a skill tune INCAR tags,
k-points, and ENCUT. Output parsing uses ``Vasprun`` for energy/convergence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DFTConfig
from .base import Engine, PrepareResult, looks_like_cif

# calc_type -> pymatgen input-set class name (resolved lazily to avoid a hard
# import at module load).
_SET_FOR_CALC = {
    "relax": "MPRelaxSet",
    "static": "MPStaticSet",
    "single_point": "MPStaticSet",
}


class VaspEngine(Engine):
    name = "vasp"
    calc_types = ("relax", "static", "single_point")

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
        from pymatgen.core import Structure
        from pymatgen.io.vasp import sets as vasp_sets

        overrides = overrides or {}
        warnings: List[str] = []
        if calc_type not in self.calc_types:
            warnings.append(
                f"calc_type '{calc_type}' is not a runnable VASP template "
                f"(supported: {', '.join(self.calc_types)}). Falling back to 'relax'. "
                "Capabilities like DOS, band structure, phonons, or AIMD must be "
                "expressed via `overrides` (INCAR/KPOINTS settings) on a supported "
                "template, or by extending this engine adapter — do NOT assume the "
                "intended physics ran."
            )
            calc_type = "relax"

        # --- parse structure -------------------------------------------------
        fmt = structure_format
        if fmt == "auto":
            fmt = "cif" if looks_like_cif(structure) else "poscar"
        struct = Structure.from_str(structure, fmt=fmt)
        if not struct.is_ordered:
            warnings.append(
                "Structure is disordered (partial occupancies); order it first "
                "(e.g. pymatgen_majority_orderer) before a production VASP run."
            )

        # --- build the input set --------------------------------------------
        set_name = _SET_FOR_CALC.get(calc_type, "MPRelaxSet")
        set_cls = getattr(vasp_sets, set_name)
        user_incar = dict(overrides.get("incar", {}))
        if "encut" in overrides:
            user_incar["ENCUT"] = overrides["encut"]

        set_kwargs: Dict[str, Any] = {"user_incar_settings": user_incar}
        if "kpts" in overrides:
            # A single int -> reciprocal density; a list -> explicit grid.
            kpts = overrides["kpts"]
            if isinstance(kpts, (list, tuple)):
                set_kwargs["user_kpoints_settings"] = {"grid_density": None, "kpoints": [list(kpts)]}
            else:
                set_kwargs["user_kpoints_settings"] = {"reciprocal_density": kpts}

        vis = set_cls(struct, **set_kwargs)

        # --- write inputs (POTCAR may be unavailable in dev environments) ----
        Path(workdir).mkdir(parents=True, exist_ok=True)
        pp_path = overrides.get("vasp_pp_path")
        if pp_path:
            os.environ["PMG_VASP_PSP_DIR"] = pp_path
            os.environ["VASP_PP_PATH"] = pp_path

        input_files: Dict[str, str] = {}
        try:
            vis.write_input(workdir)
            input_files = {
                name: str(Path(workdir) / name)
                for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
                if (Path(workdir) / name).is_file()
            }
        except Exception as exc:  # typically missing pseudopotentials
            warnings.append(
                f"Full input set write failed ({exc}); wrote INCAR/POSCAR/KPOINTS "
                "without POTCAR. Set engines.vasp_pp_path before a real run."
            )
            vis.incar.write_file(str(Path(workdir) / "INCAR"))
            vis.poscar.write_file(str(Path(workdir) / "POSCAR"))
            try:
                vis.kpoints.write_file(str(Path(workdir) / "KPOINTS"))
            except Exception:
                pass
            input_files = {
                name: str(Path(workdir) / name)
                for name in ("INCAR", "POSCAR", "KPOINTS")
                if (Path(workdir) / name).is_file()
            }

        resolved = {
            "calc_type": calc_type,
            "input_set": set_name,
            "incar": {k: v for k, v in dict(vis.incar).items()},
            "formula": struct.composition.reduced_formula,
            "n_sites": len(struct),
            "input_files": list(input_files.keys()),
            "output_file": "vasprun.xml",
        }
        return PrepareResult(
            input_files=input_files,
            resolved_params=resolved,
            output_file="vasprun.xml",
            warnings=warnings,
        )

    def run_commands(
        self, record_workdir: str, resolved_params: Dict[str, Any],
        resources: Dict[str, Any], config: DFTConfig,
    ) -> List[str]:
        from string import Template

        ntasks = resources.get("ntasks", config.scheduler.default_ntasks)
        cmd = Template(config.engines.vasp_command).safe_substitute(ntasks=ntasks)
        prefix = []
        if config.engines.vasp_pp_path:
            prefix.append(f'export VASP_PP_PATH="{config.engines.vasp_pp_path}"')
        return prefix + [cmd]

    def parse_results(
        self, workdir: str, calc_type: str, resolved_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        from pymatgen.io.vasp.outputs import Vasprun

        vasprun_path = Path(workdir) / "vasprun.xml"
        if not vasprun_path.is_file():
            return {
                "parsed": False,
                "error": "vasprun.xml not found; the calculation may not have run.",
            }
        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
        except Exception as exc:
            return {"parsed": False, "error": f"Failed to parse vasprun.xml: {exc}"}

        final = vr.final_structure
        from pymatgen.io.cif import CifWriter

        return {
            "parsed": True,
            "converged": bool(vr.converged),
            "converged_electronic": bool(vr.converged_electronic),
            "converged_ionic": bool(vr.converged_ionic),
            "final_energy_eV": float(vr.final_energy),
            "n_ionic_steps": len(vr.ionic_steps),
            "formula": final.composition.reduced_formula,
            "final_structure_cif": str(CifWriter(final)),
            "output_files": [
                name for name in ("vasprun.xml", "OUTCAR", "CONTCAR", "OSZICAR")
                if (Path(workdir) / name).is_file()
            ],
        }

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

        # Continue from the relaxed geometry where available.
        contcar = src / "CONTCAR"
        if contcar.is_file() and contcar.stat().st_size > 0:
            shutil.copy2(contcar, dst / "POSCAR")
            copied["POSCAR"] = str(dst / "POSCAR")
        elif (src / "POSCAR").is_file():
            shutil.copy2(src / "POSCAR", dst / "POSCAR")
            copied["POSCAR"] = str(dst / "POSCAR")
            warnings.append("No usable CONTCAR; restarted from the original POSCAR.")

        for name in ("INCAR", "KPOINTS", "POTCAR"):
            f = src / name
            if f.is_file():
                shutil.copy2(f, dst / name)
                copied[name] = str(dst / name)

        resolved = dict(resolved_params)
        resolved["input_files"] = list(copied.keys())
        return PrepareResult(
            input_files=copied,
            resolved_params=resolved,
            output_file=resolved.get("output_file", "vasprun.xml"),
            warnings=warnings,
        )
