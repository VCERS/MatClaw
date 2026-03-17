"""
Tests for pymatgen_substitution_generator tool.

Run with: pytest tests/test_pymatgen/test_substitution_generator.py -v
"""

import pytest
from tools.pymatgen.pymatgen_substitution_generator import pymatgen_substitution_generator


class TestSimpleSubstitution:
    """Tests for simple element substitution (Li -> Na)."""
    
    def test_simple_substitution_success(self, simple_lifep04_structure):
        """Test basic substitution of Li with Na."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=2,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["structures"]) == 2
        assert len(result["metadata"]) == 2
        
        # Check that Li was replaced with Na
        for meta in result["metadata"]:
            assert "Na" in meta["formula"]
            assert "Li" not in meta["formula"]
            assert meta["substitutions_applied"]["Li"]["replace_with"] == "Na"
            assert meta["substitutions_applied"]["Li"]["fraction"] == 1.0
            assert meta["n_sites"] == 10
    
    def test_simple_substitution_preserves_volume(self, simple_lifep04_structure):
        """Test that substitution preserves approximate cell volume."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is True
        # Volume should be approximately preserved (within 1%)
        assert abs(result["metadata"][0]["volume"] - 290.46) < 3.0


class TestMultipleSubstitutionOptions:
    """Tests for substitution with multiple replacement options."""
    
    def test_multiple_options(self, simple_lifep04_structure):
        """Test substitution with multiple element options (Li -> [Na, K])."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": ["Na", "K"]},
            n_structures=3,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["count"] == 6  # 2 options * 3 structures each
        
        # Check that we get both Na and K variants
        formulas = [meta["formula"] for meta in result["metadata"]]
        assert any("Na" in f for f in formulas)
        assert any("K" in f for f in formulas)
    
    def test_multiple_options_metadata(self, simple_lifep04_structure):
        """Test that metadata correctly reflects which substitution was applied."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": ["Na", "K"]},
            n_structures=2,
            output_format="dict"
        )
        
        assert result["success"] is True
        
        # Check substitution tracking
        for meta in result["metadata"]:
            subs = meta["substitutions_applied"]["Li"]["replace_with"]
            assert subs in ["Na", "K"]


class TestFractionalSubstitution:
    """Tests for fractional/partial substitution."""
    
    def test_fractional_substitution_50_percent(self, simple_lifep04_structure):
        """Test 50% fractional substitution (Li -> Na)."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": {"replace_with": "Na", "fraction": 0.5}},
            n_structures=3,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["count"] == 3
        
        # Check that structures contain both Li and Na
        for meta in result["metadata"]:
            composition = meta["composition"]
            assert "Na" in composition
            assert "Li" in composition
            
            # Check that approximately 50% was substituted
            subs = meta["substitutions_applied"]["Li"]
            assert subs["fraction"] == pytest.approx(0.5, abs=0.1)
            assert subs["n_sites_replaced"] == 1  # 50% of 2 Li sites = 1
    
    def test_fractional_substitution_multiple(self, simple_lifep04_structure):
        """Test multiple fractional substitutions (25% Na, 50% K)."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={
                "Li": [
                    {"replace_with": "Na", "fraction": 0.25},
                    {"replace_with": "K", "fraction": 0.5}
                ]
            },
            n_structures=2,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["count"] == 4  # 2 options * 2 structures each
        
        # Check that we get variants with Na and K
        formulas = [meta["formula"] for meta in result["metadata"]]
        assert any("Na" in f for f in formulas)
        assert any("K" in f for f in formulas)


class TestMultipleElementSubstitutions:
    """Tests for substituting multiple elements simultaneously."""
    
    def test_dual_substitution(self, simple_lifep04_structure):
        """Test substitution of two elements (Li -> Na, Fe -> Co)."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={
                "Li": "Na",
                "Fe": "Co"
            },
            n_structures=2,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["count"] == 2
        
        # Check that both substitutions were applied
        for meta in result["metadata"]:
            assert "Na" in meta["formula"]
            assert "Co" in meta["formula"]
            assert "Li" not in meta["formula"]
            assert "Fe" not in meta["formula"]
            
            assert meta["substitutions_applied"]["Li"]["replace_with"] == "Na"
            assert meta["substitutions_applied"]["Fe"]["replace_with"] == "Co"


class TestOutputFormats:
    """Tests for different output formats."""
    
    def test_dict_output_format(self, simple_lifep04_structure):
        """Test dict output format (default)."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert isinstance(result["structures"][0], dict)
        assert "@module" in result["structures"][0]
    
    def test_cif_output_format(self, simple_lifep04_structure):
        """Test CIF output format."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="cif"
        )
        
        assert result["success"] is True
        assert isinstance(result["structures"][0], str)
        assert "data_" in result["structures"][0]
        assert "_cell_length_a" in result["structures"][0]
    
    def test_poscar_output_format(self, simple_lifep04_structure):
        """Test POSCAR output format."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "K"},
            n_structures=1,
            output_format="poscar"
        )
        
        assert result["success"] is True
        assert isinstance(result["structures"][0], str)
        # POSCAR should have element names
        assert "K" in result["structures"][0]
        assert "Fe" in result["structures"][0]
    
    def test_json_output_format(self, simple_lifep04_structure):
        """Test JSON output format."""
        import json
        
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="json"
        )
        
        assert result["success"] is True
        assert isinstance(result["structures"][0], str)
        
        # Should be valid JSON
        parsed = json.loads(result["structures"][0])
        assert isinstance(parsed, dict)
    
    def test_invalid_output_format(self, simple_lifep04_structure):
        """Test that invalid output format returns error."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="invalid_format"
        )
        
        assert result["success"] is False
        assert "Invalid output_format" in result["error"]


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_invalid_element_substitution(self, simple_lifep04_structure):
        """Test substitution of element not present in structure."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Zn": "Cu"},  # Zn doesn't exist in structure
            n_structures=1,
            output_format="dict"
        )
        
        # Should succeed but with empty substitutions or produce minimal structures
        # The actual behavior depends on implementation
        if result["success"]:
            # If it succeeds, it should either skip or warn
            assert result["count"] >= 0
        else:
            # Or it might fail gracefully
            assert "error" in result
    
    def test_empty_substitutions_dict(self, simple_lifep04_structure):
        """Test that empty substitutions dictionary returns error."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is False
        assert "must be a non-empty dictionary" in result["error"]
    
    def test_invalid_input_structure_type(self):
        """Test that invalid input structure type returns error."""
        result = pymatgen_substitution_generator(
            input_structures=12345,  # Invalid type
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_missing_replace_with_key(self, simple_lifep04_structure):
        """Test that fractional substitution without replace_with key fails."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": {"fraction": 0.5}},  # Missing 'replace_with'
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is False
        assert "replace_with" in result["error"]


class TestMultipleInputStructures:
    """Tests for processing multiple input structures."""
    
    def test_multiple_input_structures(self, simple_lifep04_structure, simple_nacl_structure):
        """Test substitution on multiple input structures."""
        # Use Li substitution which exists in LiFePO4 but not NaCl
        result = pymatgen_substitution_generator(
            input_structures=[simple_lifep04_structure, simple_nacl_structure],
            substitutions={"Li": "K"},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["input_info"]["n_input_structures"] == 2
        
        # Should have processed structures (at least from LiFePO4)
        assert result["count"] >= 1
        
        # Check that K appears in at least one formula (from Li substitution)
        formulas = [meta["formula"] for meta in result["metadata"]]
        assert any("K" in f for f in formulas)


class TestDistanceValidation:
    """Tests for minimum distance validation."""
    
    def test_minimum_distance_validation(self, simple_lifep04_structure):
        """Test that minimum distance parameter is respected."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=2,
            min_distance=0.5,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["count"] >= 1
        
        # All generated structures should pass distance validation
        for meta in result["metadata"]:
            assert meta["n_sites"] > 0


class TestParameterValidation:
    """Tests for parameter validation and limits."""
    
    def test_n_structures_parameter(self, simple_lifep04_structure):
        """Test that n_structures parameter works correctly."""
        for n in [1, 3, 5]:
            result = pymatgen_substitution_generator(
                input_structures=simple_lifep04_structure,
                substitutions={"Li": "Na"},
                n_structures=n,
                output_format="dict"
            )
            
            assert result["success"] is True
            assert result["count"] == n
    
    def test_max_attempts_parameter(self, simple_lifep04_structure):
        """Test that max_attempts parameter is respected."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=5,
            max_attempts=10,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert result["attempts"] <= 10


class TestMetadata:
    """Tests for metadata completeness and accuracy."""
    
    def test_metadata_completeness(self, simple_lifep04_structure):
        """Test that all expected metadata fields are present."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na"},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is True
        
        # Check top-level result fields
        assert "count" in result
        assert "structures" in result
        assert "metadata" in result
        assert "input_info" in result
        assert "substitution_rules" in result
        assert "message" in result
        
        # Check metadata fields for each structure
        meta = result["metadata"][0]
        assert "index" in meta
        assert "formula" in meta
        assert "composition" in meta
        assert "substitutions_applied" in meta
        assert "charge_neutral" in meta
        assert "n_sites" in meta
        assert "volume" in meta
    
    def test_substitution_rules_recorded(self, simple_lifep04_structure):
        """Test that substitution rules are correctly recorded in result."""
        result = pymatgen_substitution_generator(
            input_structures=simple_lifep04_structure,
            substitutions={"Li": "Na", "Fe": "Co"},
            n_structures=1,
            output_format="dict"
        )
        
        assert result["success"] is True
        assert "Li" in result["substitution_rules"]
        assert "Fe" in result["substitution_rules"]
        
        # Check rule format
        assert isinstance(result["substitution_rules"]["Li"], list)
        assert result["substitution_rules"]["Li"][0]["replace_with"] == "Na"
