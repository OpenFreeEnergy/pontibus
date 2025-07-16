# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import json

import pytest
from gufe import SmallMoleculeComponent
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
from gufe.tokenization import JSON_HANDLER
from numpy.testing import assert_allclose
from openff.toolkit import Molecule
from openff.units import unit

from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.utils.molecules import WATER


@pytest.fixture
def CO_offmol():
    solvent_offmol = Molecule.from_smiles("CO")
    solvent_offmol.generate_conformers(n_conformers=1)
    solvent_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    return solvent_offmol


@pytest.fixture
def CO_component(CO_offmol):
    s = ExtendedSolventComponent(solvent_molecule=SmallMoleculeComponent.from_openff(CO_offmol))
    return s


def test_defaults():
    s = ExtendedSolventComponent()

    assert s.smiles == "[H][O][H]"
    assert s.positive_ion == "Na+"
    assert s.negative_ion == "Cl-"
    assert s.ion_concentration == 0.0 * unit.molar
    assert not s.neutralize
    assert s.solvent_molecule == WATER


def test_neq_different_smc():
    water_off = WATER.to_openff()
    # Create a water with partial charges
    water_off.assign_partial_charges(partial_charge_method="gasteiger")
    WATER2 = SmallMoleculeComponent.from_openff(water_off)
    s1 = ExtendedSolventComponent(solvent_molecule=WATER)
    s2 = ExtendedSolventComponent(solvent_molecule=WATER2)

    assert s1 != s2
    assert s1.smiles == "[H][O][H]" == s2.smiles


def test_neq_different_solvent():
    meth_off = Molecule.from_smiles("C")
    meth_off.generate_conformers()
    METH = SmallMoleculeComponent.from_openff(meth_off)
    s1 = ExtendedSolventComponent()
    s2 = ExtendedSolventComponent(solvent_molecule=METH)

    assert s1 != s2
    assert s1.smiles == "[H][O][H]"
    assert s2.smiles == "[H][C]([H])([H])[H]"
    assert s1.smiles != s2.smiles


def test_dict_roundtrip_eq(CO_component, CO_offmol):
    s2 = ExtendedSolventComponent.from_dict(CO_component.to_dict())
    assert CO_component == s2
    assert CO_component.solvent_molecule == s2.solvent_molecule
    assert_allclose(
        CO_component.solvent_molecule.to_openff().partial_charges,
        CO_offmol.partial_charges,
    )
    # Smiles isn't a dict entry, so make sure it got preserved
    assert CO_component.smiles == s2.smiles


def test_json_serializable_eq(CO_component, CO_offmol):
    CO_component_json = json.dumps(CO_component.to_dict(), cls=JSON_HANDLER.encoder)
    CO_component_dict = json.loads(CO_component_json, cls=JSON_HANDLER.decoder)

    s2 = ExtendedSolventComponent.from_dict(CO_component_dict)

    assert CO_component == s2
    assert CO_component.solvent_molecule == s2.solvent_molecule
    assert_allclose(
        CO_component.solvent_molecule.to_openff().partial_charges,
        CO_offmol.partial_charges,
    )
    # Smiles isn't a dict entry, so make sure it got preserved
    assert CO_component.smiles == s2.smiles


def test_keyedchain_json_serializable_eq(CO_component, CO_offmol):
    CO_component_json = json.dumps(CO_component.to_keyed_chain(), cls=JSON_HANDLER.encoder)
    CO_component_keyed_chain = json.loads(CO_component_json, cls=JSON_HANDLER.decoder)

    s2 = ExtendedSolventComponent.from_keyed_chain(CO_component_keyed_chain)

    assert CO_component == s2
    assert CO_component.solvent_molecule == s2.solvent_molecule
    assert_allclose(
        CO_component.solvent_molecule.to_openff().partial_charges,
        CO_offmol.partial_charges,
    )
    # Smiles isn't a dict entry, so make sure it got preserved
    assert CO_component.smiles == s2.smiles


def test_keyed_dict_roundtrip_eq():
    s1 = ExtendedSolventComponent()
    s2 = ExtendedSolventComponent.from_keyed_dict(s1.to_keyed_dict())
    assert s1 == s2
    # Smiles isn't a dict entry, so make sure it got preserved
    assert s1.smiles == s2.smiles
    # Check the smcs
    assert s1.solvent_molecule == s2.solvent_molecule
    assert isinstance(s1.solvent_molecule, SmallMoleculeComponent)


def test_shallow_dict_roundtrip_eq():
    s1 = ExtendedSolventComponent()
    s2 = ExtendedSolventComponent.from_shallow_dict(s1.to_shallow_dict())
    assert s1 == s2
    # Smiles isn't a dict entry, so make sure it got preserved
    assert s1.smiles == s2.smiles
    # Check the smcs
    assert s1.solvent_molecule == s2.solvent_molecule
    assert isinstance(s1.solvent_molecule, SmallMoleculeComponent)


class TestSolventComponent(GufeTokenizableTestsMixin):
    cls = ExtendedSolventComponent
    key = "ExtendedSolventComponent-f297bd89a615557b2b94d241eff240ce"
    repr = "ExtendedSolventComponent(name=[H][O][H], Na+, Cl-)"

    @pytest.fixture
    def instance(self):
        return ExtendedSolventComponent(solvent_molecule=WATER)
