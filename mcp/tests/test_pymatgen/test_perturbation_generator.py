"""
Tests for pymatgen_perturbation_generator tool.

Run with: pytest tests/test_pymatgen/test_perturbation_generator.py -v
"""

import pytest
from tools.pymatgen.pymatgen_perturbation_generator import pymatgen_perturbation_generator


class TestBasicDisplacement:
    """Tests for random atomic displacement (rattling)."""

    def test_displacement_generates_structures(self, simple_lifep04_structure):
        """Basic displacement should succeed and return requested count."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=3,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["structures"]) == 3
        assert len(result["metadata"]) == 3

    def test_displacement_preserves_composition(self, simple_lifep04_structure):
        """Perturbation must not change the chemical formula."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.2,
            n_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        for m in result["metadata"]:
            assert m["formula"] == result["metadata"][0]["formula"]

    def test_displacement_preserves_n_sites(self, simple_lifep04_structure):
        """Number of sites must remain unchanged after displacement."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        for m in result["metadata"]:
            assert m["n_sites"] == 10  # fixture has 10 atoms

    def test_zero_displacement_unchanged_volume(self, simple_lifep04_structure):
        """With displacement_max=0 and no strain, volume should be identical."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            n_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        for m in result["metadata"]:
            assert m["volume_change_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_displacement_magnitude_bound(self, simple_lifep04_structure):
        """Actual max displacement should never exceed displacement_max."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.05,
            n_structures=5,
            output_format="dict",
        )
        assert result["success"] is True
        for m in result["metadata"]:
            assert m["displacement_max_actual_ang"] <= 0.05 + 1e-9


class TestStrain:
    """Tests for lattice strain application."""

    def test_uniform_positive_strain(self, simple_lifep04_structure):
        """Positive uniform strain should increase the cell volume."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=1.0,
            n_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        for m in result["metadata"]:
            # 1% strain on each axis → volume ≈ (1.01)^3 − 1 ≈ 3.03% increase
            assert m["volume_change_pct"] == pytest.approx(3.03, abs=0.1)
            assert m["strain_applied"]["e_xx_pct"] == pytest.approx(1.0, abs=1e-4)

    def test_uniform_negative_strain(self, simple_lifep04_structure):
        """Negative uniform strain should decrease the cell volume."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=-1.0,
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["metadata"][0]["volume_change_pct"] < 0.0

    def test_strain_range(self, simple_lifep04_structure):
        """strain_percent=[min, max] should sample strains within that range."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=[-2.0, 2.0],
            n_structures=5,
            seed=42,
            output_format="dict",
        )
        assert result["success"] is True
        for m in result["metadata"]:
            pct = m["strain_applied"]["e_xx_pct"]
            assert -2.0 - 1e-6 <= pct <= 2.0 + 1e-6

    def test_voigt_strain(self, simple_lifep04_structure):
        """Six-component Voigt strain should be applied correctly."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=[1.0, 2.0, 0.5, 0.0, 0.0, 0.0],
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        sa = result["metadata"][0]["strain_applied"]
        assert sa["e_xx_pct"] == pytest.approx(1.0, abs=1e-4)
        assert sa["e_yy_pct"] == pytest.approx(2.0, abs=1e-4)
        assert sa["e_zz_pct"] == pytest.approx(0.5, abs=1e-4)

    def test_no_strain(self, simple_lifep04_structure):
        """When strain_percent=None, strain_applied metadata should be None."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=None,
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["metadata"][0]["strain_applied"] is None


class TestReproducibility:
    """Tests for seed-based reproducibility."""

    def test_same_seed_same_output(self, simple_lifep04_structure):
        """Two runs with the same seed must produce identical volumes and displacements."""
        kwargs = dict(
            input_structures=simple_lifep04_structure,
            displacement_max=0.15,
            strain_percent=[-1.0, 1.0],
            n_structures=3,
            seed=123,
            output_format="dict",
        )
        r1 = pymatgen_perturbation_generator(**kwargs)
        r2 = pymatgen_perturbation_generator(**kwargs)
        assert r1["success"] and r2["success"]
        for m1, m2 in zip(r1["metadata"], r2["metadata"]):
            assert m1["volume"] == pytest.approx(m2["volume"], rel=1e-9)
            assert m1["displacement_rms_ang"] == pytest.approx(m2["displacement_rms_ang"], rel=1e-9)

    def test_different_seeds_different_output(self, simple_lifep04_structure):
        """Different seeds should (almost certainly) produce different structures."""
        r1 = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.15,
            n_structures=1,
            seed=1,
            output_format="dict",
        )
        r2 = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.15,
            n_structures=1,
            seed=99,
            output_format="dict",
        )
        assert r1["success"] and r2["success"]
        # Volumes are the same (no strain) but displacements differ
        assert r1["metadata"][0]["displacement_rms_ang"] != pytest.approx(
            r2["metadata"][0]["displacement_rms_ang"], rel=1e-6
        )


class TestOutputFormats:
    """Tests for all supported output formats."""

    def test_dict_output(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, dict)
        assert "@module" in s

    def test_cif_output(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=1,
            output_format="cif",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, str)
        assert "data_" in s
        assert "_cell_length_a" in s

    def test_poscar_output(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=1,
            output_format="poscar",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, str)
        assert "Li" in s

    def test_json_output(self, simple_lifep04_structure):
        import json
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=1,
            output_format="json",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)


class TestMultipleInputStructures:
    """Tests for multiple input structures."""

    def test_two_inputs_multiplies_count(self, simple_lifep04_structure, simple_nacl_structure):
        """n input structures x n_structures = total count."""
        result = pymatgen_perturbation_generator(
            input_structures=[simple_lifep04_structure, simple_nacl_structure],
            displacement_max=0.1,
            n_structures=3,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["count"] == 6
        assert result["input_info"]["n_input_structures"] == 2

    def test_source_structure_label_in_metadata(self, simple_lifep04_structure, simple_nacl_structure):
        """metadata should record which source structure each variant came from."""
        result = pymatgen_perturbation_generator(
            input_structures=[simple_lifep04_structure, simple_nacl_structure],
            displacement_max=0.1,
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        source_labels = {m["source_structure"] for m in result["metadata"]}
        assert len(source_labels) == 2  # one label per input structure


class TestErrorHandling:
    """Tests for error handling and parameter validation."""

    def test_invalid_output_format(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=1,
            output_format="xyz",
        )
        assert result["success"] is False
        assert "Invalid output_format" in result["error"]

    def test_invalid_input_type(self):
        result = pymatgen_perturbation_generator(
            input_structures=12345,
            displacement_max=0.1,
            n_structures=1,
        )
        assert result["success"] is False
        assert "error" in result

    def test_strain_range_wrong_order(self, simple_lifep04_structure):
        """[max, min] (reversed) should return an error."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=[2.0, -2.0],
            n_structures=1,
        )
        assert result["success"] is False
        assert "error" in result

    def test_strain_list_wrong_length(self, simple_lifep04_structure):
        """A list with neither 2 nor 6 elements should error."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.0,
            strain_percent=[1.0, 2.0, 3.0],
            n_structures=1,
        )
        assert result["success"] is False
        assert "error" in result


class TestMetadata:
    """Tests for metadata completeness."""

    def test_top_level_fields(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        for key in ("count", "structures", "metadata", "input_info", "perturbation_params", "message"):
            assert key in result

    def test_per_structure_metadata_fields(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            strain_percent=0.5,
            n_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        m = result["metadata"][0]
        for key in (
            "index", "source_structure", "variant", "formula", "n_sites",
            "volume", "volume_change_pct", "displacement_max_actual_ang",
            "displacement_rms_ang", "strain_applied", "symmetry_restored",
        ):
            assert key in m, f"Missing metadata key: {key}"

    def test_perturbation_params_recorded(self, simple_lifep04_structure):
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.05,
            strain_percent=1.0,
            n_structures=2,
            seed=7,
            output_format="dict",
        )
        assert result["success"] is True
        pp = result["perturbation_params"]
        assert pp["displacement_max_ang"] == 0.05
        assert pp["strain_percent"] == 1.0
        assert pp["seed"] == 7
        assert pp["n_structures_per_input"] == 2

    def test_variant_index_increments(self, simple_lifep04_structure):
        """metadata 'variant' field should go 1, 2, 3, … per source structure."""
        result = pymatgen_perturbation_generator(
            input_structures=simple_lifep04_structure,
            displacement_max=0.1,
            n_structures=4,
            output_format="dict",
        )
        assert result["success"] is True
        variants = [m["variant"] for m in result["metadata"]]
        assert variants == [1, 2, 3, 4]
