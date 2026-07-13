"""
Analysis tools test fixtures.
"""

import pytest
import os


@pytest.fixture
def simple_nacl_structure():
    """Simple NaCl rock salt structure for testing (CIF format)."""
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(5.64)
    species = ["Na", "Cl"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords).to(fmt="cif")


@pytest.fixture
def overlapping_atoms_structure():
    """Structure with two atoms too close together (CIF format)."""
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(5.0)
    species = ["Na", "Na"]
    coords = [[0, 0, 0], [0.05, 0, 0]]  # Only 0.25 Å apart
    return Structure(lattice, species, coords).to(fmt="cif")


@pytest.fixture
def charged_structure():
    """Structure that is not charge neutral (CIF format)."""
    from pymatgen.core import Structure, Lattice, Species
    
    lattice = Lattice.cubic(5.0)
    # Two Na+ and no anion = net +2 charge
    species = [Species("Na", 1), Species("Na", 1)]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords).to(fmt="cif")


@pytest.fixture
def valid_licoo2_structure():
    """Valid LiCoO2 structure (CIF format)."""
    from pymatgen.core import Structure, Lattice
    
    # Layered LiCoO2 structure
    lattice = Lattice.from_parameters(2.82, 2.82, 14.05, 90, 90, 120)
    species = ["Li", "Co", "O", "O", "O", "O"]
    coords = [
        [0, 0, 0],
        [0, 0, 0.5],
        [0, 0, 0.25],
        [0, 0, 0.75],
        [0.333, 0.667, 0.25],
        [0.667, 0.333, 0.75],
    ]
    return Structure(lattice, species, coords).to(fmt="cif")


@pytest.fixture
def high_coordination_structure():
    """Structure with unusually high coordination number (CIF format)."""
    from pymatgen.core import Structure, Lattice
    
    # Small lattice with many atoms = high coordination
    lattice = Lattice.cubic(3.0)
    species = ["Fe"] * 10
    coords = [
        [0, 0, 0],
        [0.3, 0, 0],
        [0, 0.3, 0],
        [0, 0, 0.3],
        [0.3, 0.3, 0],
        [0.3, 0, 0.3],
        [0, 0.3, 0.3],
        [0.3, 0.3, 0.3],
        [0.15, 0.15, 0.15],
        [0.45, 0.45, 0.45],
    ]
    return Structure(lattice, species, coords).to(fmt="cif")


# Materials Project API key fixtures
@pytest.fixture
def mp_api_key():
    """Materials Project API key from environment variable."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        pytest.skip("MP_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def mp_api_key_available():
    """Check if Materials Project API key is available."""
    return bool(os.environ.get("MP_API_KEY"))


@pytest.fixture
def ordered_bgse_with_integer_occupancy():
    """
    Ordered BaGa4Se7 CIF with integer occupancy values (e.g. '1' not '1.0').
    
    This triggers pymatgen to use numpy.longlong internally, which was causing
    JSON serialization failures in the MCP server. Reproduces the bug from:
    tests/test-2.8(5)/candidates/ordered/CAND-011_ordered.cif
    """
    return """# generated using pymatgen
data_BaGa4Se7
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   7.62520000
_cell_length_b   6.51140000
_cell_length_c   14.70200000
_cell_angle_alpha   90.00000000
_cell_angle_beta   121.24000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   BaGa4Se7
_chemical_formula_sum   'Ba2 Ga8 Se14'
_cell_volume   624.12182096
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Ba  Ba0  1  0.56392000  0.35352000  0.04634000  1.0
  Ba  Ba1  1  0.56392000  0.64648000  0.54634000  1.0
  Ga  Ga2  1  0.00720000  0.16590000  0.74082000  1
  Ga  Ga3  1  0.00720000  0.83410000  0.24082000  1
  Ga  Ga4  1  0.00000000  0.00720000  0.99999000  1
  Ga  Ga5  1  0.00000000  0.99280000  0.49999000  1
  Ga  Ga6  1  0.49270000  0.15830000  0.72796000  1
  Ga  Ga7  1  0.49270000  0.84170000  0.22796000  1
  Ga  Ga8  1  0.23740000  0.36320000  0.22517000  1
  Ga  Ga9  1  0.23740000  0.63680000  0.72517000  1
  Se  Se1  1  0.78240000  0.15060000  0.55319000  1.0
  Se  Se1  1  0.78240000  0.84940000  0.05319000  1.0
  Se  Se2  1  0.00590000  0.36330000  0.03880000  1.0
  Se  Se2  1  0.00590000  0.63670000  0.53880000  1.0
  Se  Se3  1  0.32850000  0.10990000  0.54215000  1.0
  Se  Se3  1  0.32850000  0.89010000  0.04215000  1.0
  Se  Se4  1  0.32940000  0.01780000  0.30967000  1.0
  Se  Se4  1  0.32940000  0.98220000  0.80967000  1.0
  Se  Se5  1  0.09730000  0.50800000  0.32329000  1.0
  Se  Se5  1  0.09730000  0.49200000  0.82329000  1.0
  Se  Se6  1  0.57030000  0.49480000  0.28064000  1.0
  Se  Se6  1  0.57030000  0.50520000  0.78064000  1.0
  Se  Se7  1  0.81690000  0.03070000  0.30402000  1.0
  Se  Se7  1  0.81690000  0.96930000  0.80402000  1.0
"""
