"""
Tests for using MACE foundation models as a backend for the matcalc tools.

MACE names are resolved by ``matcalc.load_fp`` through ``matcalc.utils.MODEL_REGISTRY``
(canonical names) and ``MODEL_ALIASES`` (short names). Every ``matcalc_calc_*`` tool
routes its ``calculator`` argument through ``load_fp``, so any MACE name works as a
drop-in backend once ``mace-torch`` is installed — no per-tool code change required.

Two layers of tests:

* ``TestMaceRegistry`` — offline wiring checks (package import + registry entries).
  These run anywhere and prove the MACE names the tools advertise are recognized.
* ``TestMaceBackendIntegration`` — actually load and run a MACE model. The weights are
  downloaded on first use; if they can't be fetched (no network and nothing cached),
  the ``mace_calc`` fixture calls ``pytest.skip`` so these don't hard-fail in a
  sandbox. With weights available, they exercise the real model end-to-end.

Run with:        pytest tests/matcalc/test_mace_backend.py -v
Offline only:    pytest tests/matcalc/test_mace_backend.py -v -m "not slow"
Full (network):  pytest tests/matcalc/test_mace_backend.py -v -m slow
"""

import pytest

# Smallest MACE-MP foundation model -> smallest download for the integration tests.
MACE_TEST_MODEL = "MACE-MP-0-small"

# 2-atom Si primitive cell (POSCAR) — small enough for a fast force evaluation.
SI_POSCAR = """Si2
1.0
3.348898 0.000000 1.933487
1.116299 3.157372 1.933487
0.000000 0.000000 3.866975
Si
2
direct
0.750000 0.750000 0.750000 Si
0.250000 0.250000 0.250000 Si"""


# --------------------------------------------------------------------------- #
# Offline: package + registry wiring (no model download)
# --------------------------------------------------------------------------- #
class TestMaceRegistry:
    """The MACE backend is wired up correctly even without network access."""

    def test_mace_torch_installed(self):
        """mace-torch must be importable for the MACE backend to work."""
        pytest.importorskip("mace", reason="mace-torch not installed")
        from mace.calculators import MACECalculator, mace_mp, mace_off  # noqa: F401

    def test_registry_contains_mace_models(self):
        """matcalc's foundation-potential registry recognizes the MACE families."""
        from matcalc.utils import MODEL_REGISTRY

        expected = [
            "MACE-MP-0-small",
            "MACE-MP-0-medium",
            "MACE-MP-0-large",
            "MACE-MPA-0-medium",
            "MACE-OMAT-0-medium",
            "MACE-MatPES-PBE-0",
            "MACE-MatPES-r2SCAN-0",
        ]
        missing = [name for name in expected if name not in MODEL_REGISTRY]
        assert not missing, f"MACE models missing from MODEL_REGISTRY: {missing}"

    def test_mace_short_alias_resolves(self):
        """The convenience alias 'mace' resolves to the MPA-0 medium canonical name."""
        from matcalc.utils import MODEL_ALIASES

        assert MODEL_ALIASES.get("mace") == "MACE-MPA-0-medium"


# --------------------------------------------------------------------------- #
# Integration: actually load and run a MACE model (downloads weights on 1st use)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def mace_calc():
    """Load the smallest MACE-MP model once, skipping if the weights can't be fetched.

    Loading downloads the model on first use. In an offline environment with no cached
    weights this raises, so we skip the dependent tests instead of failing them.
    """
    pytest.importorskip("mace", reason="mace-torch not installed")
    import matcalc as mtc

    try:
        return mtc.load_fp(MACE_TEST_MODEL)
    except Exception as exc:  # noqa: BLE001 - download/SSL/offline all mean "skip"
        pytest.skip(f"Could not load {MACE_TEST_MODEL} (no network / no cached weights): {exc}")


@pytest.mark.slow
@pytest.mark.integration
class TestMaceBackendIntegration:
    """End-to-end checks that a MACE model runs as a matcalc backend."""

    def test_load_fp_returns_calculator(self, mace_calc):
        """load_fp on a MACE name returns a usable ASE calculator."""
        from ase.calculators.calculator import Calculator

        assert isinstance(mace_calc, Calculator)

    def test_single_point_energy_and_forces(self, mace_calc):
        """MACE computes a finite energy and correctly shaped forces for bulk Si."""
        import numpy as np
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        si.calc = mace_calc

        energy = si.get_potential_energy()
        forces = si.get_forces()

        assert np.isfinite(energy)
        assert forces.shape == (len(si), 3)
        assert np.all(np.isfinite(forces))

    def test_matcalc_md_with_mace_backend(self, mace_calc):
        """matcalc_calc_md drives a short MD run using a MACE calculator name."""
        from tools.matcalc.matcalc_calc_md import matcalc_calc_md

        result = matcalc_calc_md(
            structure_input=SI_POSCAR,
            calculator=MACE_TEST_MODEL,
            ensemble="nvt",
            temperature=300.0,
            steps=2,  # minimal — just confirm the backend is driven
            relax_structure=False,
        )

        assert result["success"] is True, result.get("error")
        assert result["energy"] is not None
        assert MACE_TEST_MODEL.upper() in result["calculator"].upper()

    def test_matcalc_eos_with_mace_backend(self, mace_calc):
        """matcalc_calc_eos fits an equation of state using a MACE backend."""
        from tools.matcalc.matcalc_calc_eos import matcalc_calc_eos

        result = matcalc_calc_eos(
            input_structure=SI_POSCAR,
            calculator=MACE_TEST_MODEL,
            relax_structure=True,
            fmax=0.1,
        )

        assert result["success"] is True, result.get("error")
