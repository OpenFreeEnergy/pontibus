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
    key = "ASFEProtocol-798d96f939ae6898c385e31e48caae6d"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestSolventUnit(GufeTokenizableTestsMixin):
    cls = ASFESolventUnit
    repr = "ASFESolventUnit(Absolute Solvation, benzene solvent leg: repeat 2 generation 0)"
    key = None

    @pytest.fixture()
    def instance(self, solvent_protocol_unit):
        return solvent_protocol_unit

    def test_key_stable(self):
        pytest.skip()


class TestVacuumUnit(GufeTokenizableTestsMixin):
    cls = ASFEVacuumUnit
    repr = (
        "ASFEVacuumUnit(Absolute Solvation, benzene vacuum leg: repeat 2 generation 0)"
    )
    key = None

    @pytest.fixture()
    def instance(self, vacuum_protocol_unit):
        return vacuum_protocol_unit

    def test_key_stable(self):
        pytest.skip()


class TestProtocolResult(GufeTokenizableTestsMixin):
    cls = ASFEProtocolResult
    key = "ASFEProtocolResult-f1172ed96a55d778bdfcc8d9ce0299f2"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result