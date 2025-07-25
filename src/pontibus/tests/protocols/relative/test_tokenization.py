# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json

import gufe
import openfe
from openfe import SolventComponent
import pytest
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin

from pontibus.protocols.relative import (
    HybridTopProtocol,
    HybridTopProtocolResult,
    HybridTopProtocolUnit,
)


@pytest.fixture
def protocol():
    return HybridTopProtocol(HybridTopProtocol.default_settings())


@pytest.fixture
def vacuum_protocol():
    settings = HybridTopProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = 'nocutoff'
    return HybridTopProtocol(settings=settings)


@pytest.fixture
def vacuum_protocol_unit(
    vacuum_protocol,
    benzene_to_toluene_mapping,
    benzene_vacuum_system,
    toluene_vacuum_system
):
    pus = vacuum_protocol.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping=benzene_to_toluene_mapping,
    )
    return list(pus.protocol_units)[0]


@pytest.fixture
def solvent_protocol_unit(
    protocol,
    benzene_to_toluene_mapping,
    benzene_system,
    toluene_system
):
    pus = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping=benzene_to_toluene_mapping,
    )
    return list(pus.protocol_units)[0]


@pytest.fixture
def protocol_result(rfe_solv_transformation_json):
    d = json.loads(rfe_solv_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder)
    pr = HybridTopProtocolResult.from_dict(d["protocol_result"])
    return pr


class TestProtocol(GufeTokenizableTestsMixin):
    cls = HybridTopProtocol
    key = None
    repr = "HybridTopProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestSolventUnit(GufeTokenizableTestsMixin):
    cls = HybridTopProtocolUnit
    repr = "HybridTopProtocolUnit("
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
    cls = HybridTopProtocolUnit
    repr = "HybridTopProtocolUnit("
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
    cls = HybridTopProtocolResult
    key = None
    repr = "HybridTopProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result

    def test_repr(self, instance):
        """
        Overwrites the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)
