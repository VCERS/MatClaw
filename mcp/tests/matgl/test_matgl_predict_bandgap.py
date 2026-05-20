"""
Tests for matgl_predict_bandgap tool.

Run with: pytest tests/matgl/test_matgl_predict_bandgap.py -v
"""

import pytest
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from tools.matgl.matgl_predict_bandgap import matgl_predict_bandgap
import dgl # needs to be imported even if not used directly


class TestMatglPredictBandgap:
    """Tests for ML band gap prediction."""

    def test_basic_prediction_with_dict_input(self):
        """Test basic band gap prediction with dict input."""
        from pymatgen.core import Lattice, Structure
        
        # Create a simple CsCl structure (should be wide band gap)
        struct = Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(4.1437),
            ["Cs", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        
        result = matgl_predict_bandgap(
            input_structure=str(CifWriter(struct)),
            model="MEGNet-BandGap-mfi-MP-2019.4.1"
        )
        
        # Check basic success
        assert result["success"] is True
        assert "band_gap" in result
        assert "model_used" in result
        assert result["model_used"] == "MEGNet-BandGap-mfi-MP-2019.4.1"
        
        # Check that we got a reasonable band gap value
        bandgap = result["band_gap"]
        assert isinstance(bandgap, float)
        assert bandgap >= 0, "Band gap should be non-negative"
        # CsCl should be an insulator with large band gap
        assert bandgap > 2.0, "CsCl should have a wide band gap"
        
        # Check metadata
        assert result["formula"] == "CsCl"
        assert result["num_sites"] == 2
        assert "material_class" in result

    def test_metallic_structure(self):
        """Test with a metallic structure (zero band gap)."""
        from pymatgen.core import Lattice, Structure
        
        # Create a simple Cu structure (FCC metal)
        struct = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(3.61),
            ["Cu"],
            [[0, 0, 0]]
        )
        
        result = matgl_predict_bandgap(
            input_structure=str(CifWriter(struct))
        )
        
        assert result["success"] is True
        bandgap = result["band_gap"]
        # Metals should have very small or zero band gap
        assert bandgap < 0.5, "Cu should be metallic with small/zero band gap"
        assert "Metal" in result["material_class"] or "Narrow" in result["material_class"]

    def test_semiconductor_structure(self):
        """Test with a semiconductor structure."""
        from pymatgen.core import Lattice, Structure
        
        # Create a simple GaAs structure (zinc blende semiconductor)
        struct = Structure.from_spacegroup(
            "F-43m",
            Lattice.cubic(5.65),
            ["Ga", "As"],
            [[0, 0, 0], [0.25, 0.25, 0.25]]
        )
        
        result = matgl_predict_bandgap(
            input_structure=str(CifWriter(struct))
        )
        
        assert result["success"] is True
        bandgap = result["band_gap"]
        # GaAs has a direct band gap, typically better predicted than Si
        assert bandgap >= 0, f"GaAs should have non-negative band gap, got {bandgap}"
        assert bandgap < 3.0, f"GaAs band gap should be reasonable, got {bandgap}"
        # Check it's classified as some type of semiconductor (not insulator)
        assert "Semiconductor" in result["material_class"] or "gap" in result["material_class"].lower()

    def test_different_structures(self):
        """Test prediction for different structure types with various band gaps."""
        from pymatgen.core import Lattice, Structure
        
        structures = [
            # NaCl - wide band gap insulator
            Structure.from_spacegroup(
                "Fm-3m",
                Lattice.cubic(5.64),
                ["Na", "Cl"],
                [[0, 0, 0], [0.5, 0.5, 0.5]]
            ),
            # GaAs - narrow/medium band gap semiconductor
            Structure.from_spacegroup(
                "F-43m",
                Lattice.cubic(5.65),
                ["Ga", "As"],
                [[0, 0, 0], [0.25, 0.25, 0.25]]
            ),
        ]
        
        for struct in structures:
            result = matgl_predict_bandgap(input_structure=str(CifWriter(struct)))
            assert result["success"] is True
            assert "band_gap" in result
            assert result["band_gap"] >= 0
            print(f"{result['formula']}: {result['band_gap']:.3f} eV ({result['material_class']})")

    def test_cif_string_input(self):
        """Test with CIF string input."""
        from pymatgen.core import Lattice, Structure
        
        # Create structure and export as CIF
        struct = Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(4.1437),
            ["Cs", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        cif_string = struct.to(fmt="cif")
        
        result = matgl_predict_bandgap(input_structure=cif_string)
        
        assert result["success"] is True
        assert "band_gap" in result
        assert result["formula"] == "CsCl"

    def test_poscar_string_input(self):
        """Test with POSCAR string input."""
        from pymatgen.core import Lattice, Structure

        struct = Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(4.1437),
            ["Cs", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        poscar_string = str(Poscar(struct))

        result = matgl_predict_bandgap(input_structure=poscar_string)

        assert result["success"] is True
        assert "band_gap" in result
        assert result["formula"] == "CsCl"
        assert result["num_sites"] == 2

    def test_material_classification(self):
        """Test that material classification is provided."""
        from pymatgen.core import Lattice, Structure
        
        struct = Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(4.1437),
            ["Cs", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        
        result = matgl_predict_bandgap(input_structure=str(CifWriter(struct)))
        
        assert result["success"] is True
        assert "material_class" in result
        assert "interpretation" in result
        # Verify classification is one of the expected types
        valid_classes = [
            "Metal/Conductor",
            "Narrow Band Gap Semiconductor",
            "Semiconductor",
            "Wide Band Gap Semiconductor",
            "Very Wide Band Gap Semiconductor/Insulator"
        ]
        assert result["material_class"] in valid_classes

    def test_structure_info_included(self):
        """Test that structure information is included in result."""
        from pymatgen.core import Lattice, Structure
        
        struct = Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(4.1437),
            ["Cs", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        
        result = matgl_predict_bandgap(input_structure=str(CifWriter(struct)))
        
        assert result["success"] is True
        assert "structure_info" in result
        info = result["structure_info"]
        assert "formula" in info
        assert "num_sites" in info
        assert "volume" in info
        assert "density_g_per_cm3" in info
        assert info["num_sites"] == 2
        assert info["formula"] == "CsCl"

    def test_different_functionals(self):
        """Test band gap prediction with different DFT functionals."""
        from pymatgen.core import Lattice, Structure
        
        # Use GaAs as test case (well-known semiconductor, ~1.4 eV experimental)
        struct = Structure.from_spacegroup(
            "F-43m",
            Lattice.cubic(5.65),
            ["Ga", "As"],
            [[0, 0, 0], [0.25, 0.25, 0.25]]
        )
        
        functionals = ["PBE", "GLLB-SC", "HSE", "SCAN"]
        results = {}
        
        for functional in functionals:
            result = matgl_predict_bandgap(
                input_structure=str(CifWriter(struct)),
                functional=functional
            )
            
            assert result["success"] is True, f"Failed with functional {functional}"
            assert "band_gap" in result
            assert "functional" in result
            assert result["functional"] == functional
            results[functional] = result["band_gap"]
            print(f"{functional}: {result['band_gap']:.4f} eV ({result['material_class']})")
        
        # GLLB-SC should give reasonable value for GaAs (~0.9-1.1 eV for DFT-level)
        assert 0.5 < results["GLLB-SC"] < 1.5, \
            f"GLLB-SC should give reasonable GaAs band gap, got {results['GLLB-SC']}"
        
        # PBE typically gives much lower values (may even be negative/near-zero)
        assert results["PBE"] < results["GLLB-SC"], \
            "PBE should give lower band gap than GLLB-SC"
        
        # All functionals should produce different results
        assert len(set(results.values())) > 1, \
            "Different functionals should produce different predictions"

    def test_default_functional_is_gllb_sc(self):
        """Test that GLLB-SC is the default functional."""
        from pymatgen.core import Lattice, Structure
        
        struct = Structure.from_spacegroup(
            "F-43m",
            Lattice.cubic(5.65),
            ["Ga", "As"],
            [[0, 0, 0], [0.25, 0.25, 0.25]]
        )
        
        # Call without specifying functional
        result_default = matgl_predict_bandgap(input_structure=str(CifWriter(struct)))
        
        # Call with explicit GLLB-SC
        result_explicit = matgl_predict_bandgap(
            input_structure=str(CifWriter(struct)),
            functional="GLLB-SC"
        )
        
        assert result_default["success"] is True
        assert result_explicit["success"] is True
        assert result_default["functional"] == "GLLB-SC"
        assert result_explicit["functional"] == "GLLB-SC"
        # Results should be identical
        assert abs(result_default["band_gap"] - result_explicit["band_gap"]) < 0.001

    def test_invalid_structure_handling(self):
        """Test handling of invalid structure input."""
        result = matgl_predict_bandgap(
            input_structure={"invalid": "structure"}
        )
        
        assert result["success"] is False
        assert "error" in result

    def test_band_gap_ranges(self):
        """Test that different materials fall into expected band gap ranges."""
        from pymatgen.core import Lattice, Structure
        
        # Test well-known materials that DFT-based models predict accurately
        test_cases = [
            # (structure, expected_range_min, expected_range_max, description)
            (
                Structure.from_spacegroup("F-43m", Lattice.cubic(5.65), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
                0.0, 2.5, "GaAs semiconductor"
            ),
            (
                Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                2.0, 10.0, "CsCl wide gap insulator"
            ),
        ]
        
        for struct, min_gap, max_gap, description in test_cases:
            result = matgl_predict_bandgap(input_structure=str(CifWriter(struct)))
            assert result["success"] is True
            bandgap = result["band_gap"]
            assert min_gap <= bandgap <= max_gap, (
                f"{description}: expected band gap between {min_gap}-{max_gap} eV, "
                f"got {bandgap:.3f} eV"
            )
            print(f"{description}: {bandgap:.3f} eV - {result['material_class']}")
