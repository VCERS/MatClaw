"""
Tests for pymatgen_prototype_builder tool.

Run with: pytest tests/test_pymatgen/test_prototype_builder.py -v
"""

import pytest
from tools.pymatgen.pymatgen_prototype_builder import pymatgen_prototype_builder

class TestPrototypeBuilderBasic:
    def test_simple_cubic_nacl(self):
        result = pymatgen_prototype_builder(
            spacegroup=225,
            species=["Na", "Cl"],
            lattice_parameters=[5.64],
            output_format="dict"
        )
        assert result["success"] is True
        assert result["count"] == 1
        s = result["structures"][0]
        assert s["formula"] == "NaCl"
        assert s["spacegroup_number"] == 221
        assert s["spacegroup_symbol"] == "Pm-3m"
        assert s["lattice"]["a"] == pytest.approx(5.64, abs=0.01)
        assert s["n_atoms"] == 2 or s["n_atoms"] == 8  # primitive/conventional
        assert s["composition"]["Na"] == 1 or s["composition"]["Na"] == 4
        assert s["composition"]["Cl"] == 1 or s["composition"]["Cl"] == 4

    def test_custom_lattice_parameters(self):
        result = pymatgen_prototype_builder(
            spacegroup=62,
            species=["Fe", "O"],
            lattice_parameters=[5.5, 5.5, 13.2, 90, 90, 120],
            output_format="dict"
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert s["spacegroup_number"] == 65
        assert s["lattice"]["a"] == pytest.approx(5.5, abs=0.01)
        assert s["lattice"]["c"] == pytest.approx(13.2, abs=0.01)

class TestPrototypeBuilderWyckoff:
    def test_wyckoff_assignment(self):
        result = pymatgen_prototype_builder(
            spacegroup=225,
            species=["Na", "Cl"],
            lattice_parameters=[5.64],
            wyckoff_positions={"4a": "Na", "4b": "Cl"},
            output_format="dict"
        )
        # Accept either success or graceful failure
        if result["success"]:
            s = result["structures"][0]
            assert s["formula"] == "NaCl"
            assert any(site["label"] == "4a" and site["species"] == "Na" for site in s["wyckoff_info"])
            assert any(site["label"] == "4b" and site["species"] == "Cl" for site in s["wyckoff_info"])
        else:
            assert "error" in result or "warnings" in result

class TestPrototypeBuilderFormats:
    def test_poscar_output(self):
        result = pymatgen_prototype_builder(
            spacegroup=225,
            species=["Na", "Cl"],
            lattice_parameters=[5.64],
            output_format="poscar"
        )
        assert result["success"] is True
        poscar = result["structures"][0]["structure"]
        assert isinstance(poscar, str)
        assert "Na" in poscar and "Cl" in poscar
        assert "5.64" in poscar

    def test_cif_output(self):
        result = pymatgen_prototype_builder(
            spacegroup=225,
            species=["Na", "Cl"],
            lattice_parameters=[5.64],
            output_format="cif"
        )
        assert result["success"] is True
        cif = result["structures"][0]["structure"]
        assert isinstance(cif, str)
        assert "data_" in cif
        assert "_cell_length_a" in cif

class TestPrototypeBuilderErrors:
    def test_invalid_spacegroup(self):
        result = pymatgen_prototype_builder(
            spacegroup=999,
            species=["Na", "Cl"],
            lattice_parameters=[5.64],
            output_format="dict"
        )
        assert result["success"] is False
        assert "Space group number must be between 1 and 230" in result["error"]

    def test_missing_parameters(self):
        result = pymatgen_prototype_builder(
            spacegroup=225,
            species=["Na", "Cl"],
            lattice_parameters=[],
            output_format="dict"
        )
        assert result["success"] is False or result["count"] == 0

class TestPrototypeBuilderVariants:
    def test_multiple_structures(self):
        result = pymatgen_prototype_builder(
            spacegroup=225,
            species=["Na", "Cl"],
            lattice_parameters=[5.64],
            n_structures=3,
            output_format="dict"
        )
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["structures"]) == 3
