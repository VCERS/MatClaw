"""
Tests for pymatgen_structure_editor tool.

Run with: pytest tests/pymatgen/test_structure_editor.py -v
"""

import json

from tools.pymatgen.pymatgen_structure_editor import pymatgen_structure_editor


class TestRemoveSites:
    """Tests for explicit site removal operations."""

    def test_remove_sites_by_index(self, simple_lifep04_structure):
        from pymatgen.core import Structure

        result = pymatgen_structure_editor(
            input_structures=simple_lifep04_structure,
            operations=[
                {
                    "type": "remove_sites",
                    "selection": {
                        "mode": "index",
                        "indices": [6, 7],
                        "species": ["O", "O"],
                    },
                    "label": "oxygen_divacancy",
                }
            ],
            output_format="cif",
        )

        assert result["success"] is True
        assert result["count"] == 1
        edited = Structure.from_str(result["structures"][0], fmt="cif")
        assert len(edited) == 8

        op_meta = result["metadata"][0]["operations_applied"][0]
        assert op_meta["type"] == "remove_sites"
        assert op_meta["n_sites_removed"] == 2
        assert [s["site_index_original"] for s in op_meta["sites_removed"]] == [6, 7]

    def test_remove_sites_by_nearest_fractional_coords(self, simple_nacl_structure):
        from pymatgen.core import Structure

        result = pymatgen_structure_editor(
            input_structures=simple_nacl_structure,
            operations=[
                {
                    "type": "remove_sites",
                    "selection": {
                        "mode": "nearest_to_coords",
                        "coords": [[0.02, 0.01, 0.0]],
                        "coords_are_fractional": True,
                        "species": ["Na"],
                    },
                    "label": "sodium_vacancy",
                }
            ],
            output_format="cif",
        )

        assert result["success"] is True
        edited = Structure.from_str(result["structures"][0], fmt="cif")
        assert len(edited) == 1
        assert edited[0].specie.symbol == "Cl"


class TestReplaceAndInsert:
    """Tests for replace_sites and insert_sites operations."""

    def test_replace_sites_by_index(self, simple_nacl_structure):
        from pymatgen.core import Structure

        result = pymatgen_structure_editor(
            input_structures=simple_nacl_structure,
            operations=[
                {
                    "type": "replace_sites",
                    "selection": {
                        "mode": "index",
                        "indices": [0],
                        "species": ["Na"],
                    },
                    "new_species": "K",
                    "label": "K_on_Na",
                }
            ],
            output_format="poscar",
        )

        assert result["success"] is True
        edited = Structure.from_str(result["structures"][0], fmt="poscar")
        symbols = {site.specie.symbol for site in edited}
        assert "K" in symbols
        assert "Na" not in symbols

        op_meta = result["metadata"][0]["operations_applied"][0]
        assert op_meta["sites_replaced"][0]["old_species"] == "Na"
        assert op_meta["sites_replaced"][0]["new_species"] == "K"

    def test_insert_sites(self, simple_nacl_structure):
        parsed = json.loads(
            pymatgen_structure_editor(
                input_structures=simple_nacl_structure,
                operations=[
                    {
                        "type": "insert_sites",
                        "sites": [
                            {
                                "species": "Li",
                                "coords": [0.25, 0.25, 0.25],
                                "coords_are_fractional": True,
                            }
                        ],
                        "label": "Li_interstitial",
                    }
                ],
                output_format="json",
            )["structures"][0]
        )

        assert parsed["@class"] == "Structure"
        assert len(parsed["sites"]) == 3

    def test_insert_sites_multiple(self, simple_nacl_structure):
        result = pymatgen_structure_editor(
            input_structures=simple_nacl_structure,
            operations=[
                {
                    "type": "insert_sites",
                    "sites": [
                        {
                            "species": "Li",
                            "coords": [0.25, 0.25, 0.25],
                            "coords_are_fractional": True,
                        },
                        {
                            "species": "Li",
                            "coords": [0.75, 0.75, 0.75],
                            "coords_are_fractional": True,
                        },
                    ],
                    "label": "two_Li_interstitials",
                }
            ],
            output_format="json",
        )

        assert result["success"] is True
        parsed = json.loads(result["structures"][0])
        assert len(parsed["sites"]) == 4

        op_meta = result["metadata"][0]["operations_applied"][0]
        assert op_meta["type"] == "insert_sites"
        assert op_meta["n_sites_inserted"] == 2
        assert [site["species"] for site in op_meta["sites_inserted"]] == ["Li", "Li"]


class TestValidation:
    """Tests for error handling and validation."""

    def test_nearest_to_coords_outside_tolerance_fails(self, simple_nacl_structure):
        result = pymatgen_structure_editor(
            input_structures=simple_nacl_structure,
            operations=[
                {
                    "type": "remove_sites",
                    "selection": {
                        "mode": "nearest_to_coords",
                        "coords": [[0.2, 0.2, 0.2]],
                        "coords_are_fractional": True,
                        "species": ["Na"],
                    },
                }
            ],
            selection_tolerance=0.05,
        )

        assert result["success"] is False
        assert "selection_tolerance" in result["error"]