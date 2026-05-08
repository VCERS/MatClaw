"""
Tests for pymatgen_majority_orderer tool.

Run with: pytest tests/pymatgen/test_majority_orderer.py -v

The tool converts disordered structures to ordered by replacing each
multi-species site with its dominant (highest-occupancy) species.
"""

import pytest

from tools.pymatgen.pymatgen_majority_orderer import pymatgen_majority_orderer


# Helper
def _is_ordered_cif(cif_str: str) -> bool:
    """Return True if the pymatgen Structure from CIF represents a fully ordered structure."""
    from pymatgen.core import Structure
    return Structure.from_str(cif_str, fmt="cif").is_ordered


# Basic functionality
class TestBasicFunctionality:
    """Core success / correctness tests."""

    def test_majority_ordering_succeeds(self, disordered_li_na_cl):
        """Majority ordering of a 50/50 Li/Na disordered structure should succeed."""
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["structures"]) == 1
        assert len(result["metadata"]) == 1

    def test_returned_structure_is_ordered(self, disordered_li_na_cl):
        """Returned structure must be fully ordered (no partial occupancies)."""
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        assert _is_ordered_cif(result["structures"][0]), "Structure still has partial occupancies."

    def test_cell_size_unchanged(self, disordered_li_na_cl):
        """Majority ordering should NOT expand the cell (no supercell)."""
        from pymatgen.core import Structure
        
        input_struct = Structure.from_str(disordered_li_na_cl, fmt="cif")
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        
        output_struct = Structure.from_str(result["structures"][0], fmt="cif")
        assert len(output_struct) == len(input_struct), "Cell size should not change (no supercell)."

    def test_metadata_reports_disorder(self, disordered_li_na_cl):
        """Metadata should correctly identify that structure was disordered."""
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        
        meta = result["metadata"][0]
        assert meta["was_disordered"] is True
        assert meta["sites_converted"] > 0
        assert len(meta["lost_species"]) > 0

    def test_only_one_structure_returned(self, disordered_li_na_cl):
        """Majority ordering returns exactly one structure (no enumeration)."""
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        assert result["count"] == 1, "Majority orderer should return exactly one structure."


# Output formats
class TestOutputFormats:
    """Tests for different output_format options."""

    def test_cif_format(self, disordered_li_na_cl):
        """output_format='cif' should return CIF strings."""
        result = pymatgen_majority_orderer(
            input_structures=disordered_li_na_cl,
            output_format="cif"
        )
        assert result["success"] is True
        assert isinstance(result["structures"][0], str)
        assert "data_" in result["structures"][0] or "_cell_length_a" in result["structures"][0]

    def test_poscar_format(self, disordered_li_na_cl):
        """output_format='poscar' should return POSCAR strings."""
        result = pymatgen_majority_orderer(
            input_structures=disordered_li_na_cl,
            output_format="poscar"
        )
        assert result["success"] is True
        assert isinstance(result["structures"][0], str)
        # POSCAR typically has numbers on different lines
        lines = result["structures"][0].strip().split('\n')
        assert len(lines) >= 8, "POSCAR format should have at least 8 lines."

    def test_json_format(self, disordered_li_na_cl):
        """output_format='json' should return structure dictionaries."""
        result = pymatgen_majority_orderer(
            input_structures=disordered_li_na_cl,
            output_format="json"
        )
        assert result["success"] is True
        assert isinstance(result["structures"][0], dict)
        assert "@module" in result["structures"][0]
        assert "lattice" in result["structures"][0]

    def test_invalid_format_fails(self, disordered_li_na_cl):
        """Invalid output_format should return error."""
        result = pymatgen_majority_orderer(
            input_structures=disordered_li_na_cl,
            output_format="invalid_format"
        )
        assert result["success"] is False
        assert "error" in result


# Multiple structures
class TestMultipleStructures:
    """Tests for batch processing multiple structures."""

    def test_multiple_structures(self, disordered_li_na_cl):
        """Tool should handle list of multiple structures."""
        result = pymatgen_majority_orderer(
            input_structures=[disordered_li_na_cl, disordered_li_na_cl]
        )
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["structures"]) == 2
        assert len(result["metadata"]) == 2

    def test_metadata_indices_sequential(self, disordered_li_na_cl):
        """Metadata indices should be 1-based and sequential."""
        result = pymatgen_majority_orderer(
            input_structures=[disordered_li_na_cl, disordered_li_na_cl]
        )
        assert result["success"] is True
        indices = [m["index"] for m in result["metadata"]]
        assert indices == [1, 2]


# Already ordered structures
class TestOrderedInput:
    """Tests for handling already-ordered structures."""

    def test_ordered_structure_with_check(self, simple_nacl_structure):
        """With check_ordered_input=True, ordered structures should trigger warning."""
        result = pymatgen_majority_orderer(
            input_structures=simple_nacl_structure,
            check_ordered_input=True
        )
        assert result["success"] is True
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert "already fully ordered" in result["warnings"][0].lower()

    def test_ordered_structure_without_check(self, simple_nacl_structure):
        """With check_ordered_input=False, ordered structures should pass through silently."""
        result = pymatgen_majority_orderer(
            input_structures=simple_nacl_structure,
            check_ordered_input=False
        )
        assert result["success"] is True
        # Could still have warnings, but the specific "already ordered" warning should be absent
        meta = result["metadata"][0]
        assert meta["was_disordered"] is False
        assert meta["sites_converted"] == 0

    def test_ordered_structure_metadata(self, simple_nacl_structure):
        """Ordered structures should have was_disordered=False in metadata."""
        result = pymatgen_majority_orderer(
            input_structures=simple_nacl_structure,
            check_ordered_input=True
        )
        assert result["success"] is True
        meta = result["metadata"][0]
        assert meta["was_disordered"] is False
        assert meta["sites_converted"] == 0
        assert meta["lost_species"] == []


# Edge cases and error handling
class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_input_fails(self):
        """Empty input list should return error."""
        result = pymatgen_majority_orderer(input_structures=[])
        assert result["success"] is False
        assert "error" in result

    def test_invalid_cif_fails(self):
        """Invalid CIF string should return error."""
        result = pymatgen_majority_orderer(input_structures="not a valid cif")
        assert result["success"] is False
        assert "error" in result

    def test_ordering_params_in_result(self, disordered_li_na_cl):
        """Result should include ordering_params documenting the method."""
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        assert "ordering_params" in result
        assert result["ordering_params"]["method"] == "majority_species_approximation"
        assert result["ordering_params"]["supercell_expansion"] is False


# Specific chemistry tests
class TestChemistry:
    """Tests for specific chemical scenarios."""

    def test_dilute_doping_removes_minority(self):
        """For dilute doping like Sr₀.₉₉Sm₀.₀₁, minority species should be removed."""
        from pymatgen.core import Structure, Lattice
        
        # Create Sr₀.₉₉Sm₀.₀₁NbO₃-like structure
        lattice = Lattice.cubic(4.0)
        structure = Structure(
            lattice,
            [{"Sr": 0.99, "Sm": 0.01}, "Nb", "O", "O", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        )
        cif_str = structure.to(fmt="cif")
        
        result = pymatgen_majority_orderer(input_structures=cif_str)
        assert result["success"] is True
        
        meta = result["metadata"][0]
        assert "Sm" in meta["lost_species"], "Sm should be removed as minority species."
        assert meta["ordered_formula"] == "SrNbO3", "Formula should be SrNbO3 after ordering."

    def test_50_50_mixture_picks_one(self, disordered_li_na_cl):
        """For 50/50 mixture, one species should be picked (arbitrary but consistent)."""
        result = pymatgen_majority_orderer(input_structures=disordered_li_na_cl)
        assert result["success"] is True
        
        meta = result["metadata"][0]
        # Either Li or Na should be kept, the other should be lost
        assert len(meta["lost_species"]) == 1
        assert meta["lost_species"][0] in ["Li", "Na"]
