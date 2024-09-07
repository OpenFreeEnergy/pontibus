from openff.toolkit import Molecule
from gufe import SmallMoleculeComponent


offmol_water = Molecule.from_dict(
    {
        "name": "",
        "atoms": [
            {
                "atomic_number": 8,
                "formal_charge": 0,
                "is_aromatic": False,
                "stereochemistry": None,
                "name": "",
                "metadata": {},
            },
            {
                "atomic_number": 1,
                "formal_charge": 0,
                "is_aromatic": False,
                "stereochemistry": None,
                "name": "",
                "metadata": {},
            },
            {
                "atomic_number": 1,
                "formal_charge": 0,
                "is_aromatic": False,
                "stereochemistry": None,
                "name": "",
                "metadata": {},
            },
        ],
        "bonds": [
            {
                "atom1": 0,
                "atom2": 1,
                "bond_order": 1,
                "is_aromatic": False,
                "stereochemistry": None,
                "fractional_bond_order": None,
            },
            {
                "atom1": 0,
                "atom2": 2,
                "bond_order": 1,
                "is_aromatic": False,
                "stereochemistry": None,
                "fractional_bond_order": None,
            },
        ],
        "properties": {},
        "conformers_unit": "angstrom",
        "conformers": [
            b"\xbfJ\xbeq\xa5\xd2;\xc6?\xd7r\xbe\x82\x10\xac\xb0\x80\x00\x00\x00\x00\x00\x00\x00\xbf\xe9\xfe~\x8bsg\xa8\xbf\xc7|WzS\x8e:\x80\x00\x00\x00\x00\x00\x00\x00?\xea\x05.'\xdc\xdc6\xbf\xc7i%\x89\xcd\xcb\x18\x00\x00\x00\x00\x00\x00\x00\x00"
        ],
        "partial_charges": None,
        "partial_charge_unit": None,
        "hierarchy_schemes": {},
    }
)

WATER = SmallMoleculeComponent.from_openff(offmol_water)
