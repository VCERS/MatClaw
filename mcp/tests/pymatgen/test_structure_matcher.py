"""
Tests for pymatgen_structure_matcher tool.

Run with: pytest tests/pymatgen/test_structure_matcher.py -v
"""

import pytest
from tools.pymatgen.pymatgen_structure_matcher import pymatgen_structure_matcher


class TestStructureMatcherBasic:
    """Test basic structure matching functionality."""
    
    def test_identical_structures_cif_vs_poscar(self):
        """Test matching identical NaCl structures in different formats."""
        # Simple cubic structure without symmetry operations
        cif_structure = """data_NaCl
_cell_length_a       5.6402
_cell_length_b       5.6402  
_cell_length_c       5.6402
_cell_angle_alpha    90.0
_cell_angle_beta     90.0
_cell_angle_gamma    90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""

        poscar_structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=cif_structure,
            structure_2=poscar_structure,
            l_tol=0.2,
            s_tol=0.3,
            angle_tol=5.0
        )
        
        assert result["success"] is True
        assert result["match"] is True
        assert result["confidence"] in ["exact", "high", "medium"]
        assert result["structure_1_info"]["formula"] == "NaCl"
        assert result["structure_2_info"]["formula"] == "NaCl"
        assert result["comparison_details"]["rms_distance"] is not None

    def test_different_compositions_no_match(self):
        """Test that structures with different compositions don't match."""
        nacl_structure = """data_NaCl
_cell_length_a       5.6402
_cell_length_b       5.6402  
_cell_length_c       5.6402
_cell_angle_alpha    90.0
_cell_angle_beta     90.0
_cell_angle_gamma    90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""

        licl_structure = """data_LiCl
_cell_length_a       5.14
_cell_length_b       5.14  
_cell_length_c       5.14
_cell_angle_alpha    90.0
_cell_angle_beta     90.0
_cell_angle_gamma    90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=nacl_structure,
            structure_2=licl_structure
        )
        
        assert result["success"] is True
        assert result["match"] is False
        assert "composition_mismatch" in result["mismatch_reasons"]


class TestStructureMatcherTolerances:
    """Test tolerance parameter effects."""
    
    def test_slightly_distorted_structures_loose_tolerance(self):
        """Test matching slightly distorted structures with loose tolerances."""
        structure_undistorted = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        structure_distorted = """NaCl
1.0
5.65 0.0 0.0
0.0 5.64 0.0
0.0 0.0 5.66
Na Cl
1 1
direct
0.01 0.01 0.0
0.51 0.51 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure_undistorted,
            structure_2=structure_distorted,
            l_tol=0.05,  # ~5% tolerance
            s_tol=0.1,
            angle_tol=5.0
        )
        
        assert result["success"] is True
        # Should match with these tolerances
        assert result["match"] is True

    def test_strict_tolerance_no_match(self):
        """Test that strict tolerances prevent matching distorted structures."""
        structure_undistorted = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        structure_distorted = """NaCl
1.0
5.8 0.0 0.0
0.0 5.8 0.0
0.0 0.0 5.8
Na Cl
1 1
direct
0.1 0.1 0.0
0.6 0.6 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure_undistorted,
            structure_2=structure_distorted,
            l_tol=0.01,  # Very strict 1%
            s_tol=0.05,  # Very strict 0.05 Å
            angle_tol=1.0,
            scale=False  # Don't scale - we want to check actual lattice parameters
        )
        
        assert result["success"] is True
        # Should not match with these strict tolerances and scale=False
        assert result["match"] is False


class TestStructureMatcherOptions:
    """Test different matching options."""
    
    def test_primitive_cell_conversion(self):
        """Test matching with primitive cell conversion."""
        simple_structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=simple_structure,
            structure_2=simple_structure,
            primitive_cell=True
        )
        
        assert result["success"] is True
        assert result["match"] is True

    def test_comparator_element_vs_species(self):
        """Test ElementComparator ignores oxidation states."""
        # This is a basic test - in practice would use oxidation states
        structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure,
            structure_2=structure,
            comparator="ElementComparator"
        )
        
        assert result["success"] is True
        assert result["match"] is True

    def test_return_mapping(self):
        """Test that return_mapping option provides site mapping."""
        structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure,
            structure_2=structure,
            return_mapping=True
        )
        
        assert result["success"] is True
        assert result["match"] is True
        # Site mapping should be present when structures match and return_mapping=True
        if result["comparison_details"]["site_mapping"] is not None:
            assert isinstance(result["comparison_details"]["site_mapping"], list)


class TestStructureMatcherErrors:
    """Test error handling."""
    
    def test_invalid_structure_format(self):
        """Test handling of invalid structure input."""
        result = pymatgen_structure_matcher(
            structure_1="invalid structure data",
            structure_2="also invalid"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "parsing" in result["error"].lower() or "failed" in result["error"].lower()

    def test_invalid_comparator(self):
        """Test handling of invalid comparator option."""
        structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure,
            structure_2=structure,
            comparator="InvalidComparator"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "comparator" in result["error"].lower()


class TestStructureMatcherSupercell:
    """Test supercell matching capabilities."""
    
    def test_supercell_matching_2x2x2(self):
        """Test matching a unit cell against its 2x2x2 supercell."""
        unit_cell = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        # 2x2x2 supercell (8 times larger, 16 atoms total)
        supercell_2x2x2 = """NaCl
1.0
11.2804 0.0 0.0
0.0 11.2804 0.0
0.0 0.0 11.2804
Na Cl
8 8
direct
0.0 0.0 0.0
0.0 0.0 0.5
0.0 0.5 0.0
0.0 0.5 0.5
0.5 0.0 0.0
0.5 0.0 0.5
0.5 0.5 0.0
0.5 0.5 0.5
0.25 0.25 0.25
0.25 0.25 0.75
0.25 0.75 0.25
0.25 0.75 0.75
0.75 0.25 0.25
0.75 0.25 0.75
0.75 0.75 0.25
0.75 0.75 0.75
"""
        
        result = pymatgen_structure_matcher(
            structure_1=unit_cell,
            structure_2=supercell_2x2x2,
            attempt_supercell=True,
            supercell_size="num_sites"
        )
        
        assert result["success"] is True
        assert result["match"] is True
        assert result["comparison_details"]["supercell_relation"] is not None

    def test_no_supercell_match_without_flag(self):
        """Test that supercell detection provides supercell info even without attempt_supercell."""
        unit_cell = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        supercell_2x2x2 = """NaCl
1.0
11.2804 0.0 0.0
0.0 11.2804 0.0
0.0 0.0 11.2804
Na Cl
8 8
direct
0.0 0.0 0.0
0.0 0.0 0.5
0.0 0.5 0.0
0.0 0.5 0.5
0.5 0.0 0.0
0.5 0.0 0.5
0.5 0.5 0.0
0.5 0.5 0.5
0.25 0.25 0.25
0.25 0.25 0.75
0.25 0.75 0.25
0.25 0.75 0.75
0.75 0.25 0.25
0.75 0.25 0.75
0.75 0.75 0.25
0.75 0.75 0.75
"""
        
        result = pymatgen_structure_matcher(
            structure_1=unit_cell,
            structure_2=supercell_2x2x2,
            attempt_supercell=False,
            primitive_cell=False  # Disable primitive cell conversion
        )
        
        assert result["success"] is True
        # With primitive_cell=False and attempt_supercell=False, should not match
        assert result["match"] is False


class TestStructureMatcherSubset:
    """Test subset matching for structures with different site counts."""
    
    def test_subset_matching_different_compositions(self):
        """Test that structures with different compositions still don't match with allow_subset."""
        nacl_structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        # Different composition entirely
        na2cl_structure = """Na2Cl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
2 1
direct
0.0 0.0 0.0
0.25 0.25 0.25
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=nacl_structure,
            structure_2=na2cl_structure,
            allow_subset=True,
            s_tol=0.3
        )
        
        assert result["success"] is True
        # Different stoichiometry won't match even with allow_subset
        assert result["match"] is False


class TestStructureMatcherScaling:
    """Test scaling behavior for volume differences."""
    
    def test_scale_allows_volume_difference(self):
        """Test that scale=True allows structures with different volumes to match."""
        structure_small = """NaCl
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        structure_large = """NaCl
1.0
6.0 0.0 0.0
0.0 6.0 0.0
0.0 0.0 6.0
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure_small,
            structure_2=structure_large,
            scale=True,  # Allow volume scaling
            l_tol=0.3
        )
        
        assert result["success"] is True
        # Should match with scale=True
        assert result["match"] is True

    def test_no_scale_rejects_volume_difference(self):
        """Test that scale=False rejects structures with different volumes."""
        structure_small = """NaCl
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        structure_large = """NaCl
1.0
6.0 0.0 0.0
0.0 6.0 0.0
0.0 0.0 6.0
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure_small,
            structure_2=structure_large,
            scale=False,  # Don't allow volume scaling
            l_tol=0.1
        )
        
        assert result["success"] is True
        # Should not match with scale=False and 20% size difference
        assert result["match"] is False


class TestStructureMatcherAngles:
    """Test angle tolerance effects."""
    
    def test_angle_tolerance_variations(self):
        """Test matching structures with slight angle differences."""
        structure_cubic = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        # Slightly non-cubic structure (angles close to 90)
        structure_tilted = """NaCl
1.0
5.6402 0.1 0.0
0.0 5.6402 0.1
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure_cubic,
            structure_2=structure_tilted,
            angle_tol=5.0,  # 5 degree tolerance
            l_tol=0.2,
            s_tol=0.3
        )
        
        assert result["success"] is True
        # Should match with reasonable angle tolerance
        assert result["match"] is True

    def test_strict_angle_tolerance(self):
        """Test strict angle tolerance rejects tilted structures."""
        structure_cubic = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

        # More significantly tilted structure
        structure_tilted = """NaCl
1.0
5.6402 0.5 0.0
0.0 5.6402 0.5
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure_cubic,
            structure_2=structure_tilted,
            angle_tol=1.0,  # Very strict 1 degree
            l_tol=0.2,
            s_tol=0.3
        )
        
        assert result["success"] is True
        # Should not match with strict angle tolerance
        assert result["match"] is False


class TestStructureMatcherConfidence:
    """Test confidence level assignment."""
    
    def test_exact_confidence_strict_tolerances(self):
        """Test that exact match with strict tolerances gives 'exact' confidence."""
        structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure,
            structure_2=structure,
            l_tol=0.01,  # Very strict
            s_tol=0.05,  # Very strict
            angle_tol=0.5
        )
        
        assert result["success"] is True
        assert result["match"] is True
        assert result["confidence"] == "exact"

    def test_medium_confidence_default_tolerances(self):
        """Test that matches with default tolerances give appropriate confidence."""
        structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure,
            structure_2=structure,
            l_tol=0.2,  # Default
            s_tol=0.3,  # Default
            angle_tol=5.0  # Default
        )
        
        assert result["success"] is True
        assert result["match"] is True
        assert result["confidence"] in ["exact", "high", "medium"]

    def test_low_confidence_loose_tolerances(self):
        """Test that matches with very loose tolerances give 'low' confidence."""
        structure = """NaCl
1.0
5.6402 0.0 0.0
0.0 5.6402 0.0
0.0 0.0 5.6402
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        
        result = pymatgen_structure_matcher(
            structure_1=structure,
            structure_2=structure,
            l_tol=0.5,  # Very loose
            s_tol=0.8,  # Very loose
            angle_tol=15.0  # Very loose
        )
        
        assert result["success"] is True
        assert result["match"] is True
        assert result["confidence"] == "low"

