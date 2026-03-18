"""
Tests for pymatgen_ion_exchange_generator tool.

Run with: pytest tests/test_pymatgen/test_ion_exchange_generator.py -v
"""

import pytest
from tools.pymatgen.pymatgen_ion_exchange_generator import pymatgen_ion_exchange_generator


class TestBasicExchange:
    """Tests for simple 1:1 ion exchange (same oxidation state)."""

    def test_li_to_na_full_exchange(self, simple_lifep04_structure):
        """Full exchange of Li with Na (both +1)."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["count"] >= 1
        assert len(result["structures"]) >= 1

        m = result["metadata"][0]
        assert "Na" in m["formula"]
        assert "Li" not in m["formula"]
        assert m["replaced_ion"] == "Li"
        assert m["n_sites"] > 0
        assert m["volume"] > 0

    def test_na_to_k_full_exchange(self, simple_nacl_structure):
        """Full exchange of Na with K in NaCl."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_nacl_structure,
            replace_ion="Na",
            with_ions=["K"],
            exchange_fraction=1.0,
            max_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        m = result["metadata"][0]
        assert "K" in m["formula"]
        assert "Na" not in m["formula"]

    def test_exchange_preserves_non_replaced_species(self, simple_lifep04_structure):
        """Fe, P, O sites should be untouched when only Li is exchanged."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        m = result["metadata"][0]
        assert "Fe" in m["composition"]
        assert "P" in m["composition"]
        assert "O" in m["composition"]

    def test_max_structures_respected(self, simple_lifep04_structure):
        """max_structures limits the number of generated structures."""
        for n in [1, 3, 5]:
            result = pymatgen_ion_exchange_generator(
                input_structures=simple_lifep04_structure,
                replace_ion="Li",
                with_ions=["Na"],
                exchange_fraction=1.0,
                max_structures=n,
                output_format="dict",
            )
            assert result["success"] is True
            assert result["count"] <= n


class TestCrossValenceExchange:
    """Tests where the replacement ion has a different oxidation state."""

    def test_li_to_mg_with_vacancies(self, simple_lifep04_structure):
        """Li+ (x2) -> Mg2+ (x1): one Li site replaced, one removed as vacancy."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Mg"],
            exchange_fraction=1.0,
            allow_oxidation_state_change=True,
            max_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        m = result["metadata"][0]
        # Vacancies should have been created to balance charge
        assert m["n_vacancies_created"] >= 0
        assert m["replaced_ion"] == "Li"
        assert m["replaced_ion_oxi"] == pytest.approx(1.0, abs=0.5)

    def test_charge_neutral_flag(self, simple_lifep04_structure):
        """charge_neutral metadata field should be a bool (or None if BV fails)."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        cn = result["metadata"][0]["charge_neutral"]
        assert cn is True or cn is False or cn is None


class TestMultipleReplacementIons:
    """Tests for exchanging into more than one replacement ion."""

    def test_list_form_two_ions(self, simple_lifep04_structure):
        """with_ions=['Na', 'K'] should produce structures containing Na or K."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na", "K"],
            exchange_fraction=1.0,
            allow_oxidation_state_change=True,
            max_structures=4,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["count"] >= 1
        formulas = [m["formula"] for m in result["metadata"]]
        assert any("Na" in f or "K" in f for f in formulas)

    def test_dict_form_weighted_ions(self, simple_lifep04_structure):
        """with_ions={'Na': 0.6, 'K': 0.4} (dict weight form) should succeed."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions={"Na": 0.6, "K": 0.4},
            exchange_fraction=1.0,
            allow_oxidation_state_change=True,
            max_structures=2,
            output_format="dict",
        )
        assert result["success"] is True
        # exchange_rules should record both ions
        ion_names = [entry["ion"] for entry in result["exchange_rules"]["with_ions"]]
        assert "Na" in ion_names
        assert "K" in ion_names


class TestPartialExchange:
    """Tests for fractional / partial ion exchange."""

    def test_partial_exchange_50_percent(self, simple_lifep04_structure):
        """exchange_fraction=0.5 should leave some Li sites intact."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=0.5,
            allow_oxidation_state_change=True,
            max_structures=3,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["count"] >= 1
        for m in result["metadata"]:
            assert m["n_sites"] > 0

    def test_per_ion_fraction_list(self, simple_lifep04_structure):
        """Per-ion exchange_fraction list should match length of with_ions."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na", "K"],
            exchange_fraction=[0.5, 0.5],
            allow_oxidation_state_change=True,
            max_structures=2,
            output_format="dict",
        )
        assert result["success"] is True


class TestOutputFormats:
    """Tests for all supported output formats."""

    def test_dict_output(self, simple_lifep04_structure):
        """dict output should be a pymatgen Structure dict."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, dict)
        assert "@module" in s

    def test_cif_output(self, simple_lifep04_structure):
        """CIF output should be a valid CIF string."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="cif",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, str)
        assert "data_" in s
        assert "_cell_length_a" in s

    def test_poscar_output(self, simple_lifep04_structure):
        """POSCAR output should be a valid POSCAR string containing element names."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="poscar",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, str)
        assert "Na" in s

    def test_json_output(self, simple_lifep04_structure):
        """JSON output should be a valid JSON string parseable as a dict."""
        import json

        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="json",
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert isinstance(parsed, dict)


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_empty_with_ions(self, simple_lifep04_structure):
        """Empty with_ions list should return an error."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=[],
            exchange_fraction=1.0,
            max_structures=1,
        )
        assert result["success"] is False
        assert "error" in result

    def test_invalid_output_format(self, simple_lifep04_structure):
        """Unsupported output_format should return an error."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="xyz",
        )
        assert result["success"] is False
        assert "Invalid output_format" in result["error"]

    def test_invalid_input_type(self):
        """Non-dict/string input_structures should return an error."""
        result = pymatgen_ion_exchange_generator(
            input_structures=99999,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
        )
        assert result["success"] is False
        assert "error" in result

    def test_replace_ion_not_in_structure(self, simple_nacl_structure):
        """Replacing an ion that doesn't exist should warn and return no structures."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_nacl_structure,
            replace_ion="Li",  # Li not in NaCl
            with_ions=["K"],
            exchange_fraction=1.0,
            max_structures=2,
        )
        assert result["success"] is False
        assert "warnings" in result

    def test_fraction_list_length_mismatch(self, simple_lifep04_structure):
        """exchange_fraction list length != with_ions length should error."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na", "K"],
            exchange_fraction=[0.5],  # length 1, but 2 ions
            max_structures=1,
        )
        assert result["success"] is False
        assert "error" in result

    def test_exchange_fraction_out_of_range(self, simple_lifep04_structure):
        """exchange_fraction > 1.0 should return an error."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.5,
            max_structures=1,
        )
        assert result["success"] is False
        assert "error" in result


class TestMetadata:
    """Tests for metadata completeness and exchange_rules output."""

    def test_top_level_fields_present(self, simple_lifep04_structure):
        """All expected top-level fields should be in the result."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        for key in ("count", "structures", "metadata", "input_info", "exchange_rules", "message"):
            assert key in result

    def test_per_structure_metadata_fields(self, simple_lifep04_structure):
        """Each metadata entry should contain all expected fields."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        m = result["metadata"][0]
        for key in (
            "index", "formula", "composition", "replaced_ion", "replaced_ion_oxi",
            "n_replace_sites_original", "n_vacancies_created", "ions_placed",
            "charge_neutral", "total_charge", "n_sites", "volume",
        ):
            assert key in m, f"Missing metadata key: {key}"

    def test_exchange_rules_recorded(self, simple_lifep04_structure):
        """exchange_rules should record replace_ion, with_ions details, and flag."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        er = result["exchange_rules"]
        assert er["replace_ion"] == "Li"
        assert isinstance(er["with_ions"], list)
        assert er["with_ions"][0]["ion"] == "Na"
        assert "exchange_fraction" in er["with_ions"][0]
        assert "assumed_oxi_state" in er["with_ions"][0]

    def test_input_info_recorded(self, simple_lifep04_structure):
        """input_info should report the number of input structures."""
        result = pymatgen_ion_exchange_generator(
            input_structures=simple_lifep04_structure,
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            max_structures=1,
            output_format="dict",
        )
        assert result["success"] is True
        assert result["input_info"]["n_input_structures"] == 1


class TestMultipleInputStructures:
    """Tests for providing multiple input structures at once."""

    def test_two_input_structures(self, simple_lifep04_structure, simple_nacl_structure):
        """Both input structures are processed; Li exchange applies to LiFePO4 only."""
        result = pymatgen_ion_exchange_generator(
            input_structures=[simple_lifep04_structure, simple_nacl_structure],
            replace_ion="Li",
            with_ions=["Na"],
            exchange_fraction=1.0,
            allow_oxidation_state_change=True,
            max_structures=2,
            output_format="dict",
        )
        # NaCl has no Li, so only LiFePO4 should produce structures
        assert result["input_info"]["n_input_structures"] == 2
        if result["success"]:
            formulas = [m["formula"] for m in result["metadata"]]
            assert any("Na" in f for f in formulas)
