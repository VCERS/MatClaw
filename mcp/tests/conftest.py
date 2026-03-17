"""
Pytest configuration and shared fixtures for MatClaw MCP tools tests.
"""

import pytest


@pytest.fixture
def simple_lifep04_structure():
    """
    Fixture providing a simple LiFePO4-like structure for testing.
    
    Returns:
        dict: Structure dictionary compatible with pymatgen Structure.from_dict()
    """
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.orthorhombic(10.3, 6.0, 4.7)
    structure = Structure(
        lattice,
        ["Li", "Li", "Fe", "Fe", "P", "P", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.0],
            [0.75, 0.75, 0.5],
            [0.1, 0.4, 0.25],
            [0.9, 0.6, 0.75],
            [0.1, 0.2, 0.25],
            [0.9, 0.8, 0.75],
            [0.3, 0.25, 0.0],
            [0.7, 0.75, 0.5]
        ]
    )
    return structure.as_dict()


@pytest.fixture
def simple_lifep04_structure_obj():
    """
    Fixture providing a simple LiFePO4-like structure as a Structure object.
    
    Returns:
        Structure: Pymatgen Structure object
    """
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.orthorhombic(10.3, 6.0, 4.7)
    structure = Structure(
        lattice,
        ["Li", "Li", "Fe", "Fe", "P", "P", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.0],
            [0.75, 0.75, 0.5],
            [0.1, 0.4, 0.25],
            [0.9, 0.6, 0.75],
            [0.1, 0.2, 0.25],
            [0.9, 0.8, 0.75],
            [0.3, 0.25, 0.0],
            [0.7, 0.75, 0.5]
        ]
    )
    return structure


@pytest.fixture
def simple_nacl_structure():
    """
    Fixture providing a simple NaCl structure for testing.
    
    Returns:
        dict: Structure dictionary compatible with pymatgen Structure.from_dict()
    """
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return structure.as_dict()


@pytest.fixture
def simple_nacl_structure_obj():
    """
    Fixture providing a simple NaCl structure as a Structure object.
    
    Returns:
        Structure: Pymatgen Structure object
    """
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return structure
