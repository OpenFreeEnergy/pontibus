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
    ASFESolventAnalysisUnit,
    ASFESolventSetupUnit,
    ASFESolventSimUnit,
    ASFEVacuumAnalysisUnit,
    ASFEVacuumSetupUnit,
    ASFEVacuumSimUnit,
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


def _get_unit(pus, pu_class):
    for pu in pus:
        if isinstance(pu, pu_class):
            return pu
    raise ValueError(f"No unit of type {pu_class} found in protocol units")


@pytest.fixture
def solvent_protocol_setup_unit(protocol_units):
    return _get_unit(protocol_units, ASFESolventSetupUnit)


@pytest.fixture
def solvent_protocol_sim_unit(protocol_units):
    return _get_unit(protocol_units, ASFESolventSimUnit)


@pytest.fixture
def solvent_protocol_analysis_unit(protocol_units):
    return _get_unit(protocol_units, ASFESolventAnalysisUnit)


@pytest.fixture
def vacuum_protocol_setup_unit(protocol_units):
    return _get_unit(protocol_units, ASFEVacuumSetupUnit)


@pytest.fixture
def vacuum_protocol_sim_unit(protocol_units):
    return _get_unit(protocol_units, ASFEVacuumSimUnit)


@pytest.fixture
def vacuum_protocol_analysis_unit(protocol_units):
    return _get_unit(protocol_units, ASFEVacuumAnalysisUnit)


@pytest.fixture
def protocol_result(afe_solv_water_transformation_json):
    d = json.loads(afe_solv_water_transformation_json, cls=gufe.tokenization.JSON_HANDLER.decoder)
    pr = ASFEProtocolResult.from_dict(d["protocol_result"])
    return pr


class ModifiedGufeTokenizableTestsMixin(GufeTokenizableTestsMixin):
    def test_repr(self, instance):
        """
        Overrides the base `test_repr` call.
        """
        assert isinstance(repr(instance), str)
        assert self.repr in repr(instance)


class TestProtocol(ModifiedGufeTokenizableTestsMixin):
    cls = ASFEProtocol
    repr = "ASFEProtocol-"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestSolventSetupUnit(ModifiedGufeTokenizableTestsMixin):
    cls = ASFESolventSetupUnit
    repr = "ASFESolventSetupUnit(ASFE Setup: benzene solvent leg"

    @pytest.fixture()
    def instance(self, solvent_protocol_setup_unit):
        return solvent_protocol_setup_unit


class TestSolventSimUnit(ModifiedGufeTokenizableTestsMixin):
    cls = ASFESolventSimUnit
    repr = "ASFESolventSimUnit(ASFE Simulation: benzene solvent leg"

    @pytest.fixture()
    def instance(self, solvent_protocol_sim_unit):
        return solvent_protocol_sim_unit


class TestSolventAnalysisUnit(ModifiedGufeTokenizableTestsMixin):
    cls = ASFESolventAnalysisUnit
    repr = "ASFESolventAnalysisUnit(ASFE Analysis: benzene solvent leg"

    @pytest.fixture()
    def instance(self, solvent_protocol_analysis_unit):
        return solvent_protocol_analysis_unit


class TestVacuumSetupUnit(ModifiedGufeTokenizableTestsMixin):
    cls = ASFEVacuumSetupUnit
    repr = "ASFEVacuumSetupUnit(ASFE Setup: benzene vacuum leg"

    @pytest.fixture()
    def instance(self, vacuum_protocol_setup_unit):
        return vacuum_protocol_setup_unit


class TestVacuumSimUnit(ModifiedGufeTokenizableTestsMixin):
    cls = ASFEVacuumSimUnit
    repr = "ASFEVacuumSimUnit(ASFE Simulation: benzene vacuum leg"

    @pytest.fixture()
    def instance(self, vacuum_protocol_sim_unit):
        return vacuum_protocol_sim_unit


class TestVacuumAnalysisUnit(ModifiedGufeTokenizableTestsMixin):
    cls = ASFEVacuumAnalysisUnit
    repr = "ASFEVacuumAnalysisUnit(ASFE Analysis: benzene vacuum leg"

    @pytest.fixture()
    def instance(self, vacuum_protocol_analysis_unit):
        return vacuum_protocol_analysis_unit


class TestProtocolResult(ModifiedGufeTokenizableTestsMixin):
    cls = ASFEProtocolResult
    repr = "ASFEProtocolResult-"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result
