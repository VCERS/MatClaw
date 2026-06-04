"""
Pytest fixtures shared across pymatgen tool tests.
"""

import pytest


@pytest.fixture
def simple_lifep04_structure():
    """
    Fixture providing a simple LiFePO4-like structure for testing.
    
    Returns:
        str: Structure in CIF format
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
    return structure.to(fmt="cif")


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
        str: Structure in CIF format
    """
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return structure.to(fmt="cif")


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


@pytest.fixture
def disordered_li_na_cl():
    """
    2-atom Li₀.₅/Na₀.₅ cation-disordered rocksalt structure (dict).

    The cation site carries 50% Li and 50% Na occupancy; the anion site is
    fully occupied by Cl.  Enumerating up to max_cell_size=2 produces a small,
    predictable set of ordered LiNaCl₂ / Li₂Cl₂ / Na₂Cl₂ approximants.

    Returns:
        str: Structure in CIF format with partial occupancy on the cation site.
    """
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        [{"Li": 0.5, "Na": 0.5}, "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    return structure.to(fmt="cif")


@pytest.fixture
def disordered_li_na_cl_obj():
    """
    Same as disordered_li_na_cl but returns the pymatgen Structure object.

    Returns:
        Structure: Pymatgen Structure object with partial cation occupancy.
    """
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(4.0)
    return Structure(
        lattice,
        [{"Li": 0.5, "Na": 0.5}, "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def ordered_cucr2se4():
    """
    Ordered CuCr₂Se₄ spinel structure for disorder generator testing.
    
    Simplified spinel-like structure with Cu on tetrahedral sites,
    Cr on octahedral sites, and Se on anion sites.
    
    Returns:
        str: Fully ordered Structure in CIF format
    """
    from pymatgen.core import Structure, Lattice
    
    # Cubic spinel-like lattice
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Cu", "Cu", "Cr", "Cr", "Cr", "Cr", "Se", "Se", "Se", "Se",
         "Se", "Se", "Se", "Se"],
        [
            [0.125, 0.125, 0.125],  # Cu tetrahedral
            [0.875, 0.875, 0.875],  # Cu tetrahedral
            [0.5, 0.5, 0.5],        # Cr octahedral
            [0.5, 0.0, 0.0],        # Cr octahedral
            [0.0, 0.5, 0.0],        # Cr octahedral
            [0.0, 0.0, 0.5],        # Cr octahedral
            [0.25, 0.25, 0.25],     # Se
            [0.75, 0.75, 0.25],     # Se
            [0.75, 0.25, 0.75],     # Se
            [0.25, 0.75, 0.75],     # Se
            [0.75, 0.75, 0.75],     # Se
            [0.25, 0.25, 0.75],     # Se
            [0.25, 0.75, 0.25],     # Se
            [0.75, 0.25, 0.25],     # Se
        ]
    )
    return structure.to(fmt="cif")


@pytest.fixture
def ordered_cucr2se4_obj():
    """
    Same as ordered_cucr2se4 but returns the pymatgen Structure object.
    
    Returns:
        Structure: Fully ordered CuCr₂Se₄ spinel Structure object
    """
    from pymatgen.core import Structure, Lattice
    
    lattice = Lattice.cubic(10.0)
    return Structure(
        lattice,
        ["Cu", "Cu", "Cr", "Cr", "Cr", "Cr", "Se", "Se", "Se", "Se",
         "Se", "Se", "Se", "Se"],
        [
            [0.125, 0.125, 0.125],
            [0.875, 0.875, 0.875],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.75, 0.75],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
        ]
    )


@pytest.fixture
def bagase_ga_mg_uniform():
    """
    Ba-Ga-Se with 10% uniform Mg on Ga sites ONLY (Ba sites are pure).

    Contains 8 equivalent cation sites each with {Ga: 0.9, Mg: 0.1},
    plus 16 Ba sites and 32 Se sites. Total Mg = 8 × 0.1 = 0.8 atoms — not integer.
    This synthetic structure tests the case where ONE site type has uniform
    fractional occupancy (produced by disorder_generator with substitution on Ga only).
    """
    from pymatgen.core import Structure, Lattice, Element
    import numpy as np

    # Monoclinic cell similar to BaGa4Se7
    lattice = Lattice.from_parameters(6.5, 13.4, 21.0, 75.7, 90.0, 90.0)

    # 56 sites: 16 Ba + 8 mixed Ga/Mg + 32 Se
    # All 8 Ga sites get {Ga: 0.9, Mg: 0.1} — uniform fractional
    species = ["Ba"] * 16
    for _ in range(8):
        species.append({Element("Ga"): 0.9, Element("Mg"): 0.1})
    species += ["Se"] * 32

    # Simplified fractional coordinates (monoclinic P1)
    np.random.seed(42)
    coords = np.random.rand(56, 3)
    # Scale to reasonable positions
    coords[:, 0] = coords[:, 0]
    coords[:, 1:3] = coords[:, 1:3]

    structure = Structure(lattice, species, coords)
    return structure.to(fmt="cif")


@pytest.fixture
def bagase_ba_ga_mg_uniform():
    """
    Real-world co-doped BaGaSe CIF (CoDope_011.06Ba_011.06Ga) with uniform
    Mg on BOTH Ba sites AND Ga sites.

    Every equivalent site of each type has the same fractional occupancy:
      - 16 Ba sites: 89% Ba + 11% Mg (uniform across all)
      - 8 Ga sites:  89% Ga + 11% Mg (uniform across all)

    Total Mg = 16×0.11 + 8×0.11 ≈ 2.65 atoms — not integer on either sublattice.
    This is the exact disorder_generator output pattern and the hardest case
    for ordering tools (dual-site uniform fractional occupancy).
    """
    return """\
# generated using pymatgen
data_Ba14.23111111Mg2.65333333Ga7.11555556Se32
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   6.54147500
_cell_length_b   13.43554900
_cell_length_c   21.03530708
_cell_angle_alpha   75.70757262
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural
Ba14.23111111Mg2.65333333Ga7.11555556Se32
_chemical_formula_sum
'Ba14.23111111 Mg2.65333333 Ga7.11555556 Se32'
_cell_volume   1791.53548229
_cell_formula_units_Z   1
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
  Ba  Ba0  1  0.23496100  0.89315600  0.21622000  0.8894444444444445
  Mg  Mg1  1  0.23496100  0.89315600  0.21622000
  0.11055555555555556
  Ba  Ba2  1  0.73496100  0.10684400  0.28378000  0.8894444444444445
  Mg  Mg3  1  0.73496100  0.10684400  0.28378000
  0.11055555555555556
  Ba  Ba4  1  0.76503900  0.10684400  0.78378000  0.8894444444444445
  Mg  Mg5  1  0.76503900  0.10684400  0.78378000
  0.11055555555555556
  Ba  Ba6  1  0.26503900  0.89315600  0.71622000  0.8894444444444445
  Mg  Mg7  1  0.26503900  0.89315600  0.71622000
  0.11055555555555556
  Ba  Ba8  1  0.74322300  0.85420600  0.04849600  0.8894444444444445
  Mg  Mg9  1  0.74322300  0.85420600  0.04849600
  0.11055555555555556
  Ba  Ba10  1  0.24322300  0.14579400  0.45150400
  0.8894444444444445
  Mg  Mg11  1  0.24322300  0.14579400  0.45150400
  0.11055555555555556
  Ba  Ba12  1  0.25677700  0.14579400  0.95150400
  0.8894444444444445
  Mg  Mg13  1  0.25677700  0.14579400  0.95150400
  0.11055555555555556
  Ba  Ba14  1  0.75677700  0.85420600  0.54849600
  0.8894444444444445
  Mg  Mg15  1  0.75677700  0.85420600  0.54849600
  0.11055555555555556
  Ba  Ba16  1  0.23719400  0.61842800  0.03798900
  0.8894444444444445
  Mg  Mg17  1  0.23719400  0.61842800  0.03798900
  0.11055555555555556
  Ba  Ba18  1  0.73719400  0.38157200  0.46201100
  0.8894444444444445
  Mg  Mg19  1  0.73719400  0.38157200  0.46201100
  0.11055555555555556
  Ba  Ba20  1  0.76280600  0.38157200  0.96201100
  0.8894444444444445
  Mg  Mg21  1  0.76280600  0.38157200  0.96201100
  0.11055555555555556
  Ba  Ba22  1  0.26280600  0.61842800  0.53798900
  0.8894444444444445
  Mg  Mg23  1  0.26280600  0.61842800  0.53798900
  0.11055555555555556
  Ba  Ba24  1  0.23154700  0.52752400  0.31235700
  0.8894444444444445
  Mg  Mg25  1  0.23154700  0.52752400  0.31235700
  0.11055555555555556
  Ba  Ba26  1  0.73154700  0.47247600  0.18764300
  0.8894444444444445
  Mg  Mg27  1  0.73154700  0.47247600  0.18764300
  0.11055555555555556
  Ba  Ba28  1  0.76845300  0.47247600  0.68764300
  0.8894444444444445
  Mg  Mg29  1  0.76845300  0.47247600  0.68764300
  0.11055555555555556
  Ba  Ba30  1  0.26845300  0.52752400  0.81235700
  0.8894444444444445
  Mg  Mg31  1  0.26845300  0.52752400  0.81235700
  0.11055555555555556
  Mg  Mg32  1  0.32310500  0.88489900  0.40182200
  0.11055555555555556
  Ga  Ga33  1  0.32310500  0.88489900  0.40182200
  0.8894444444444445
  Mg  Mg34  1  0.82310500  0.11510100  0.09817800
  0.11055555555555556
  Ga  Ga35  1  0.82310500  0.11510100  0.09817800
  0.8894444444444445
  Mg  Mg36  1  0.67689500  0.11510100  0.59817800
  0.11055555555555556
  Ga  Ga37  1  0.67689500  0.11510100  0.59817800
  0.8894444444444445
  Mg  Mg38  1  0.17689500  0.88489900  0.90182200
  0.11055555555555556
  Ga  Ga39  1  0.17689500  0.88489900  0.90182200
  0.8894444444444445
  Mg  Mg40  1  0.31687200  0.30354200  0.09983200
  0.11055555555555556
  Ga  Ga41  1  0.31687200  0.30354200  0.09983200
  0.8894444444444445
  Mg  Mg42  1  0.81687200  0.69645800  0.40016800
  0.11055555555555556
  Ga  Ga43  1  0.81687200  0.69645800  0.40016800
  0.8894444444444445
  Mg  Mg44  1  0.68312800  0.69645800  0.90016800
  0.11055555555555556
  Ga  Ga45  1  0.68312800  0.69645800  0.90016800
  0.8894444444444445
  Mg  Mg46  1  0.18312800  0.30354200  0.59983200
  0.11055555555555556
  Ga  Ga47  1  0.18312800  0.30354200  0.59983200
  0.8894444444444445
  Se  Se24  1  0.73619800  0.95791500  0.17824000  1.0
  Se  Se25  1  0.23619800  0.04208500  0.32176000  1.0
  Se  Se26  1  0.26380200  0.04208500  0.82176000  1.0
  Se  Se27  1  0.76380200  0.95791500  0.67824000  1.0
  Se  Se28  1  0.24530600  0.88370900  0.01427200  1.0
  Se  Se29  1  0.74530600  0.11629100  0.48572800  1.0
  Se  Se30  1  0.75469400  0.11629100  0.98572800  1.0
  Se  Se31  1  0.25469400  0.88370900  0.51427200  1.0
  Se  Se32  1  0.69296300  0.87256900  0.38435000  1.0
  Se  Se33  1  0.19296300  0.12743100  0.11565000  1.0
  Se  Se34  1  0.30703700  0.12743100  0.61565000  1.0
  Se  Se35  1  0.80703700  0.87256900  0.88435000  1.0
  Se  Se36  1  0.18680000  0.72857500  0.37420500  1.0
  Se  Se37  1  0.68680000  0.27142500  0.12579500  1.0
  Se  Se38  1  0.81320000  0.27142500  0.62579500  1.0
  Se  Se39  1  0.31320000  0.72857500  0.87420500  1.0
  Se  Se40  1  0.41983800  0.68788900  0.17493100  1.0
  Se  Se41  1  0.91983800  0.31211100  0.32506900  1.0
  Se  Se42  1  0.58016200  0.31211100  0.82506900  1.0
  Se  Se43  1  0.08016200  0.68788900  0.67493100  1.0
  Se  Se44  1  0.05027600  0.68672600  0.17421800  1.0
  Se  Se45  1  0.55027600  0.31327400  0.32578200  1.0
  Se  Se46  1  0.94972400  0.31327400  0.82578200  1.0
  Se  Se47  1  0.44972400  0.68672600  0.67421800  1.0
  Se  Se48  1  0.22716900  0.42065500  0.16441400  1.0
  Se  Se49  1  0.72716900  0.57934500  0.33558600  1.0
  Se  Se50  1  0.77283100  0.57934500  0.83558600  1.0
  Se  Se51  1  0.27283100  0.42065500  0.66441400  1.0
  Se  Se52  1  0.23500000  0.38921200  0.48273700  1.0
  Se  Se53  1  0.73500000  0.61078800  0.01726300  1.0
  Se  Se54  1  0.76500000  0.61078800  0.51726300  1.0
  Se  Se55  1  0.26500000  0.38921200  0.98273700  1.0
"""
