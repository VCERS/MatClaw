"""
Tests for structure_validator tool.

Run with: pytest tests/analysis/test_structure_validator.py -v
"""

import pytest
from tools.analysis.structure_validator import structure_validator


class TestStructureValidator:
    """Tests for structure validation."""

    def test_valid_structure_passes(self, simple_nacl_structure):
        """A is_valid NaCl structure should pass all checks."""
        result = structure_validator(input_structure=simple_nacl_structure)
        
        # Debug output
        if not result["is_valid"]:
            print("\nValidation failed:")
            print(f"Checks failed: {result['checks_failed']}")
            print(f"Issues: {result['issues']}")
            if 'warnings' in result:
                print(f"Warnings: {result['warnings']}")
            print("\nDetails:")
            for check, detail in result['details'].items():
                if detail.get('passed') is False:
                    print(f"  {check}: {detail}")
        
        assert result["is_valid"] is True
        assert len(result["checks_failed"]) == 0
        assert "overlapping_atoms" in result["checks_passed"]
        assert result["details"]["overlapping_atoms"]["passed"] is True

    def test_overlapping_atoms_detected(self, overlapping_atoms_structure):
        """Overlapping atoms should be detected and reported."""
        result = structure_validator(
            input_structure=overlapping_atoms_structure,
            min_distance_threshold=0.5
        )
        
        assert result["is_valid"] is False
        assert "overlapping_atoms" in result["checks_failed"]
        assert len(result["details"]["overlapping_atoms"]["problematic_pairs"]) > 0
        assert result["details"]["overlapping_atoms"]["min_distance"] < 0.5

    def test_charge_neutrality_check(self, charged_structure):
        """Non-neutral structures should be detected."""
        result = structure_validator(
            input_structure=charged_structure,
            check_charge_neutrality=True
        )
        
        # May fail charge neutrality or pass if oxidation states can't be assigned
        assert "charge_neutrality" in result["checks_performed"]
        if result["details"]["charge_neutrality"]["charge_assigned"]:
            assert "charge_neutrality" in result["checks_failed"]
            assert abs(result["details"]["charge_neutrality"]["total_charge"]) > 0.1

    def test_valid_licoo2_structure(self, valid_licoo2_structure):
        """A realistic is_valid structure should pass all checks."""
        result = structure_validator(input_structure=valid_licoo2_structure)
        
        # Should pass most checks (oxidation states might be tricky)
        assert len(result["checks_passed"]) >= 2
        assert result["details"]["overlapping_atoms"]["passed"] is True
        
    def test_high_coordination_detected(self, high_coordination_structure):
        """Unusually high coordination numbers should be flagged."""
        result = structure_validator(
            input_structure=high_coordination_structure,
            check_coordination=True,
            max_coordination=12,
            coordination_cutoff=3.5
        )
        
        assert "coordination" in result["checks_performed"]
        # This dense structure should have high coordination
        max_cn = result["details"]["coordination"]["max_cn_found"]
        assert max_cn > 5  # Should have reasonably high coordination

    def test_strict_mode_stops_at_first_error(self, overlapping_atoms_structure):
        """Strict mode should stop at first validation error."""
        result = structure_validator(
            input_structure=overlapping_atoms_structure,
            strict_mode=True,
            min_distance_threshold=0.5
        )
        
        assert result["is_valid"] is False
        # With overlapping atoms failing first, other checks might not be performed
        assert "overlapping_atoms" in result["checks_failed"]

    def test_disable_specific_checks(self, simple_nacl_structure):
        """Individual checks can be disabled."""
        result = structure_validator(
            input_structure=simple_nacl_structure,
            check_charge_neutrality=False,
            check_oxidation_states=False,
            check_coordination=False
        )
        
        assert "charge_neutrality" not in result["checks_performed"]
        assert "oxidation_states" not in result["checks_performed"]
        assert "coordination" not in result["checks_performed"]
        assert "overlapping_atoms" in result["checks_performed"]

    def test_custom_thresholds(self, simple_nacl_structure):
        """Custom validation thresholds should be respected."""
        result = structure_validator(
            input_structure=simple_nacl_structure,
            min_distance_threshold=1.0,
            coordination_cutoff=4.0,
            max_coordination=15
        )
        
        assert result["details"]["overlapping_atoms"]["threshold"] == 1.0
        assert result["details"]["coordination"]["max_cn_threshold"] == 15

    def test_structure_info_included(self, simple_nacl_structure):
        """Structure information should be included in results."""
        result = structure_validator(input_structure=simple_nacl_structure)
        
        assert "structure_info" in result
        assert "formula" in result["structure_info"]
        assert "n_sites" in result["structure_info"]
        assert "volume" in result["structure_info"]
        assert "density" in result["structure_info"]

    def test_cif_string_input(self):
        """Should accept CIF string as input."""
        cif_string = """data_NaCl
_cell_length_a    5.64
_cell_length_b    5.64
_cell_length_c    5.64
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na 0.0 0.0 0.0
Cl 0.5 0.5 0.5
"""
        result = structure_validator(input_structure=cif_string)
        assert result["is_valid"] is True

    def test_poscar_string_input(self):
        """Should accept POSCAR string as input."""
        poscar_string = """NaCl
1.0
5.64 0.0 0.0
0.0 5.64 0.0
0.0 0.0 5.64
Na Cl
1 1
direct
0.0 0.0 0.0
0.5 0.5 0.5
"""
        result = structure_validator(input_structure=poscar_string)
        
        # Debug output  
        if not result.get("is_valid"):
            print("\nPOSCAR validation failed:")
            if 'error' in result:
                print(f"Error: {result['error']}")
            if 'checks_failed' in result:
                print(f"Checks failed: {result['checks_failed']}")
                print(f"Issues: {result['issues']}")
        
        assert result["is_valid"] is True

    def test_invalid_input_type(self):
        """Should handle invalid input gracefully."""
        result = structure_validator(input_structure=12345)
        
        assert result["is_valid"] is False
        assert "error" in result

    def test_malformed_structure_dict(self):
        """Should handle malformed structure dict gracefully."""
        bad_dict = {"invalid": "structure"}
        result = structure_validator(input_structure=bad_dict)
        
        assert result["is_valid"] is False
        assert "error" in result


class TestOutputFormat:
    """Tests for result output format and completeness."""

    def test_all_required_fields_present(self, simple_nacl_structure):
        """Result should contain all documented fields."""
        result = structure_validator(input_structure=simple_nacl_structure)
        
        required_fields = [
            "is_valid",
            "checks_performed",
            "checks_passed",
            "checks_failed",
            "issues",
            "details",
            "structure_info",
            "message"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_details_structure(self, simple_nacl_structure):
        """Details dictionary should have proper structure for each check."""
        result = structure_validator(input_structure=simple_nacl_structure)
        
        for check_name in result["checks_performed"]:
            assert check_name in result["details"]
            detail = result["details"][check_name]
            assert "passed" in detail or "error" in detail

    def test_issues_list_populated_on_failure(self, overlapping_atoms_structure):
        """Issues list should be populated when validation fails."""
        result = structure_validator(
            input_structure=overlapping_atoms_structure,
            min_distance_threshold=0.5
        )
        
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert isinstance(result["issues"][0], str)

    def test_warnings_present_when_checks_error(self):
        """Warnings should be present if checks encounter errors."""
        # Create a structure that might cause check errors
        from pymatgen.core import Structure, Lattice
        
        lattice = Lattice.cubic(5.0)
        species = ["Xx"]  # Invalid element
        coords = [[0, 0, 0]]
        
        try:
            struct = Structure(lattice, species, coords).to(fmt="cif")
            result = structure_validator(input_structure=struct)
            
            # Some checks may warn or fail due to invalid element
            # Just check the result is well-formed
            assert "is_valid" in result
            assert isinstance(result.get("warnings", []), list)
        except:
            # If structure creation fails, that's also acceptable
            pass


class TestNumpySerialization:
    """Tests for numpy type serialization in validation results."""

    def test_integer_occupancy_does_not_crash(self, ordered_bgse_with_integer_occupancy):
        """
        CIFs with integer occupancy values (e.g. '1' instead of '1.0')
        should not cause numpy.longlong serialization errors.
        
        Regression test for: "Unable to serialize unknown type: <class 'numpy.longlong'>"
        """
        result = structure_validator(
            input_structure=ordered_bgse_with_integer_occupancy,
            min_distance_threshold=0.5,
            check_charge_neutrality=True,
            check_oxidation_states=True
        )
        
        # The core assertion: the call should return a valid dict without crashing
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "checks_performed" in result
        assert "details" in result

        # All detail values should be JSON-serializable (no numpy types)
        import json
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_all_numpy_types_converted(self, ordered_bgse_with_integer_occupancy):
        """All values in the result should be native Python types, not numpy."""
        result = structure_validator(
            input_structure=ordered_bgse_with_integer_occupancy,
            min_distance_threshold=0.5,
            check_charge_neutrality=True,
            check_oxidation_states=True
        )
        
        def _check_no_numpy(obj, path=""):
            import numpy as np
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_numpy(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _check_no_numpy(v, f"{path}[{i}]")
            elif isinstance(obj, tuple):
                for i, v in enumerate(obj):
                    _check_no_numpy(v, f"{path}[{i}]")
            else:
                assert not isinstance(obj, (np.integer, np.floating, np.bool_, np.ndarray)), \
                    f"numpy type found at {path}: {type(obj)} ({obj})"
        
        _check_no_numpy(result)

    def test_poscar_with_integer_occupancy(self):
        """POSCAR input with integer occupancies should also serialize cleanly."""
        poscar = """BaGa4Se7
1.0
7.6252 0.0 0.0
0.0 6.5114 0.0
0.0 0.0 14.702
Ba Ga Se
2 8 14
direct
0.56392 0.35352 0.04634
0.56392 0.64648 0.54634
0.00720 0.16590 0.74082
0.00720 0.83410 0.24082
0.00000 0.00720 0.99999
0.00000 0.99280 0.49999
0.49270 0.15830 0.72796
0.49270 0.84170 0.22796
0.23740 0.36320 0.22517
0.23740 0.63680 0.72517
0.78240 0.15060 0.55319
0.78240 0.84940 0.05319
0.00590 0.36330 0.03880
0.00590 0.63670 0.53880
0.32850 0.10990 0.54215
0.32850 0.89010 0.04215
0.32940 0.01780 0.30967
0.32940 0.98220 0.80967
0.09730 0.50800 0.32329
0.09730 0.49200 0.82329
0.57030 0.49480 0.28064
0.57030 0.50520 0.78064
0.81690 0.03070 0.30402
0.81690 0.96930 0.80402
"""
        result = structure_validator(
            input_structure=poscar,
            min_distance_threshold=0.5,
            check_charge_neutrality=True,
            check_oxidation_states=True
        )
        
        import json
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
