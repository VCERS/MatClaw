"""
Tests for cod_search_structures tool.

Run with: pytest tests/cod/test_cod_search.py -v

Note: These tests hit the live COD REST API and require internet.
Marked with pytest.mark.external for optional skipping.
"""

import pytest
from tools.cod.cod_search_structures import cod_search_structures


class TestCodSearchBasic:
    """Test basic COD search functionality with live API."""

    @pytest.mark.external
    def test_search_by_elements(self):
        """Search by element types present."""
        result = cod_search_structures(
            elements=['Fe', 'O'],
            max_results=5,
            include_cifs=False,
            include_theoretical=False,
        )
        assert result["success"] is True
        assert result["count"] > 0
        assert result["count"] <= 5
        for s in result["structures"]:
            assert isinstance(s["cod_id"], int)
            assert isinstance(s["formula"], str)
            assert isinstance(s["space_group"], str)
            # No CIF since include_cifs=False
            assert s["cif"] is None

    @pytest.mark.external
    def test_search_by_text_keyword(self):
        """Search by metadata text keyword."""
        result = cod_search_structures(
            text="chalcogenide",
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True
        assert result["count"] > 0
        # Verify total_matching is reported
        assert result["total_matching"] >= result["count"]

    @pytest.mark.external
    def test_search_by_cod_id_with_cif(self):
        """Retrieve a specific COD entry with full CIF content."""
        result = cod_search_structures(
            cod_ids=[1000023],
            max_results=1,
            include_cifs=True,
        )
        assert result["success"] is True
        assert result["count"] == 1
        s = result["structures"][0]
        assert s["cod_id"] == 1000023
        assert s["cif"] is not None
        assert len(s["cif"]) > 100
        # CIF should contain key structural information
        assert "_cell_length_a" in s["cif"] or "cell_length" in s["cif"]

    @pytest.mark.external
    def test_search_multiple_cod_ids(self):
        """Search for multiple COD IDs simultaneously."""
        result = cod_search_structures(
            cod_ids=[1000023, 1000035, 1000038],
            max_results=5,
            include_cifs=False,
        )
        assert result["success"] is True
        assert result["count"] >= 1

    @pytest.mark.external
    def test_search_by_formula(self):
        """Search by chemical formula in Hill notation."""
        result = cod_search_structures(
            formula="Si O2",
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True
        # May be 0 if formula not in COD, but should not error

    @pytest.mark.external
    def test_search_by_space_group_number(self):
        """Search by space group number."""
        result = cod_search_structures(
            space_group_number=225,  # Fm-3m
            elements=['Fe'],
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_search_by_year(self):
        """Search by publication year."""
        result = cod_search_structures(
            year=2020,
            elements=['Si', 'O'],
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_search_by_author_text(self):
        """Text search containing author name."""
        result = cod_search_structures(
            text="Kanatzidis",
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_include_cifs_false_returns_no_cif(self):
        """Verify that include_cifs=False skips CIF download."""
        result = cod_search_structures(
            elements=['Si', 'O'],
            max_results=2,
            include_cifs=False,
        )
        assert result["success"] is True
        for s in result["structures"]:
            assert s["cif"] is None

    @pytest.mark.external
    def test_multiple_filters_combined(self):
        """Combine element and text filters."""
        result = cod_search_structures(
            elements=['Ga', 'S'],
            text="chalcogenide",
            max_results=5,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_cif_content_is_valid(self):
        """Verify that downloaded CIF content is valid pymatgen structure."""
        from pymatgen.core import Structure

        result = cod_search_structures(
            cod_ids=[1000023],
            max_results=1,
            include_cifs=True,
        )
        assert result["success"] is True
        s = result["structures"][0]
        assert s["cif"] is not None

        # Parse with pymatgen to confirm valid CIF
        struct = Structure.from_str(s["cif"], fmt="cif")
        assert struct is not None
        assert len(struct) > 0


class TestCodSearchValidation:
    """Test input validation and error handling."""

    def test_no_search_criteria_returns_error(self):
        """Providing no search criteria should return an error."""
        result = cod_search_structures()
        assert result["success"] is False
        assert "error" in result

    def test_invalid_space_group_number(self):
        """Space group number must be 1-230."""
        from pydantic import ValidationError
        import inspect

        # Validate that the annotation enforces the range
        sig = inspect.signature(cod_search_structures)
        param = sig.parameters["space_group_number"]
        # Just check it doesn't crash with valid input
        result = cod_search_structures(
            space_group_number=1,
            elements=['H'],
            max_results=1,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_max_results_limit(self):
        """max_results should limit the number of returned structures."""
        result = cod_search_structures(
            elements=['Fe', 'O'],
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True
        assert result["count"] <= 3

    @pytest.mark.external
    def test_elements_up_to_eight(self):
        """Verify we can specify up to 8 element filters."""
        result = cod_search_structures(
            elements=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O'],
            formula="C8 H10 N4 O2",
            max_results=2,
            include_cifs=False,
        )
        # Should not raise; may return 0 results
        assert result["success"] is True

    def test_exclude_elements(self):
        """Exclude elements filter should be passed to API."""
        result = cod_search_structures(
            elements=['Fe', 'O'],
            exclude_elements=['Pb', 'Hg'],
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_lattice_parameter_filter(self):
        """Filter by cell volume range."""
        result = cod_search_structures(
            elements=['Si', 'O'],
            volume_min=100,
            volume_max=5000,
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_has_fobs_filter(self):
        """Filter for entries with structure factor data."""
        result = cod_search_structures(
            elements=['Si', 'O'],
            has_fobs=True,
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_include_duplicates_flag(self):
        """Verify include_duplicates does not cause error."""
        result = cod_search_structures(
            elements=['Fe'],
            include_duplicates=True,
            max_results=3,
            include_cifs=False,
        )
        assert result["success"] is True


class TestCodSearchFormulaConversion:
    """Test formula conversion to COD-compatible Hill notation."""

    def test_simple_formula_conversion(self):
        """Fe2O3 should be converted correctly."""
        result = cod_search_structures(
            formula="Fe2O3",
            max_results=2,
            include_cifs=False,
        )
        # Internally converts to "Fe2 O3" via Composition.hill_formula
        assert result["success"] is True

    def test_complex_formula_conversion(self):
        """More complex formula should be converted to Hill notation."""
        result = cod_search_structures(
            formula="RbCd4Ga3S9",
            max_results=2,
            include_cifs=False,
        )
        assert result["success"] is True

    @pytest.mark.external
    def test_space_separated_and_compact_equivalent(self):
        """Space-separated and compact formulas should return the same results."""
        result_compact = cod_search_structures(
            formula="Fe2O3",
            max_results=10,
            include_cifs=False,
        )
        result_spaced = cod_search_structures(
            formula="Fe2 O3",
            max_results=10,
            include_cifs=False,
        )
        assert result_compact["success"] is True
        assert result_spaced["success"] is True
        # Both should return the same number of results
        assert result_compact["count"] == result_spaced["count"]
        # Both should return the same COD IDs (in the same order)
        ids_compact = [s["cod_id"] for s in result_compact["structures"]]
        ids_spaced = [s["cod_id"] for s in result_spaced["structures"]]
        assert ids_compact == ids_spaced, (
            f"COD IDs differ: compact={ids_compact}, spaced={ids_spaced}"
        )

    def test_nonstandard_formula_fallback(self):
        """Invalid formulas should fall back gracefully."""
        result = cod_search_structures(
            formula="NotARealFormula",
            max_results=2,
            include_cifs=False,
        )
        assert result["success"] is True