# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from openff.toolkit import Molecule, ForceField
from pontibus.utils.molecules import WATER
from pontibus.utils.molecule_utils import (
    _check_library_charges,
    _get_num_residues,
    _set_offmol_metadata,
)


@pytest.fixture(scope="module")
def water_off():
    return WATER.to_openff()


@pytest.fixture()
def methanol():
    m = Molecule.from_smiles("CO")
    m.generate_conformers()
    m.assign_partial_charges(partial_charge_method="gasteiger")
    return m


def test_check_library_charges_pass(water_off):
    ff = ForceField("opc.offxml")
    _check_library_charges(ff, water_off)


def test_check_library_charges_fail(methanol):
    ff = ForceField("openff-2.0.0.offxml")
    with pytest.raises(ValueError, match="No library charges"):
        _check_library_charges(ff, methanol)


def test_num_residues_base(water_off):
    assert _get_num_residues(water_off) == 1


@pytest.mark.parametrize(
     "property", ["chain_id", "residue_name", "residue_number"]
)
def test_num_residues_splitprop(property):
    m = Molecule.from_smiles("C")
    _set_offmol_metadata(m, property, "A")
    m.atoms[-1].metadata[property] = "X"

    assert _get_num_residues(m) == 2


def test_num_residues_protein(T4_protein_offtop):
    assert _get_num_residues(T4_protein_offtop.molecule(0)) == 164
