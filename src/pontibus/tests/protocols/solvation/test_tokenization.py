# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json

import gufe
import openfe
import pytest
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation import (
    ASFEProtocol,
    ASFEProtocolResult,
    ASFESolventUnit,
    ASFEVacuumUnit,
)


@pytest.fixture
def protocol():
    return ASFEProtocol(ASFEProtocol.default_settings())


@pytest.fixture
def protocol_units(protocol, benzene_modifications):
    pus = protocol.create(
        stateA=openfe.ChemicalSystem(
            {
                "solute": benzene_modifications["benzene"],
                "solvent": ExtendedSolventComponent(),
            }
        ),
        stateB=openfe.ChemicalSystem({"solvent": ExtendedSolventComponent()}),
        mapping=None,
    )
    return list(pus.protocol_units)


@pytest.fixture
def solvent_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, ASFESolventUnit):
            return pu


@pytest.fixture
def vacuum_protocol_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, ASFEVacuumUnit):
            return pu


@pytest.fixture
def protocol_result(afe_solv_transformation_json):
    d = json.loads(
        afe_solv_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder
    )
    pr = ASFEProtocolResult.from_dict(d["protocol_result"])
    return pr


class TestProtocol(GufeTokenizableTestsMixin):
    cls = ASFEProtocol
    key = "ASFEProtocol-a9fe65baa34fb42a281cf9064ba9afa0"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestSolventUnit(GufeTokenizableTestsMixin):
    cls = ASFESolventUnit
    repr = "ASFESolventUnit(Absolute Solvation, benzene solvent leg"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_unit):
        return solvent_protocol_unit

    def test_key_stable(self):
        pytest.skip()

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestVacuumUnit(GufeTokenizableTestsMixin):
    cls = ASFEVacuumUnit
    repr = "ASFEVacuumUnit(Absolute Solvation, benzene vacuum leg"
    key = None

    @pytest.fixture()
    def instance(self, vacuum_protocol_unit):
        return vacuum_protocol_unit

    def test_key_stable(self):
        pytest.skip()

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestProtocolResult(GufeTokenizableTestsMixin):
    cls = ASFEProtocolResult
    key = "ASFEProtocolResult-e711f21656c3795ed9d545c326ec717a"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result
