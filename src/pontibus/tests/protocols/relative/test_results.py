# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json

import gufe
import numpy as np
import openfe
import pytest
from openff.units import unit as offunit

from pontibus.protocols.relative import HybridTopProtocolResult


class TestSolventProtocolResult:
    @pytest.fixture()
    def protocolresult(self, rfe_solv_transformation_json):
        d = json.loads(
            rfe_solv_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, rfe_solv_transformation_json):
        d = json.loads(
            rfe_solv_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = HybridTopProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(16.94, abs=0.5)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(0.2, abs=0.2)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, list)
        assert len(inds) == 3

        for e, u in inds:
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_forwards_etc(self, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, list)
        far1 = far[0]
        assert isinstance(far1, dict)

        for k in [
            "fractions",
            "forward_DGs",
            "forward_dDGs",
            "reverse_DGs",
            "reverse_dDGs",
        ]:
            assert k in far1

            if k == "fractions":
                assert isinstance(far1[k], np.ndarray)
            else:
                assert isinstance(far1[k], offunit.Quantity)
                assert far1[k].is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_frwd_reverse_none_return(self, protocolresult):
        # fetch the first result
        data = [i for i in protocolresult.data.values()][0][0]
        # set the output to None
        data.outputs["forward_and_reverse_energies"] = None

        # now fetch the analysis results and expect a warning
        wmsg = "One or more ``None`` entries were found in"
        with pytest.warns(UserWarning, match=wmsg):
            protocolresult.get_forward_and_reverse_energy_analysis()

    def test_get_overlap_matrices(self, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, list)
        assert len(ovp) == 3

        ovp1 = ovp[0]
        assert isinstance(ovp1["matrix"], np.ndarray)
        assert ovp1["matrix"].shape == (11, 11)

    def test_get_replica_transition_statistics(self, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()

        assert isinstance(rpx, list)
        assert len(rpx) == 3
        rpx1 = rpx[0]
        assert "eigenvalues" in rpx1
        assert "matrix" in rpx1
        assert rpx1["eigenvalues"].shape == (11,)
        assert rpx1["matrix"].shape == (11, 11)

    def test_equilibration_iterations(self, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, list)
        assert len(eq) == 3
        assert all(isinstance(v, float) for v in eq)

    def test_production_iterations(self, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, list)
        assert len(prod) == 3
        assert all(isinstance(v, float) for v in prod)

    def test_filenotfound_replica_states(self, protocolresult):
        errmsg = "File could not be found"

        with pytest.raises(ValueError, match=errmsg):
            protocolresult.get_replica_states()


class TestVacuumProtocolResult(TestSolventProtocolResult):
    @pytest.fixture()
    def protocolresult(self, rfe_vacuum_transformation_json):
        d = json.loads(
            rfe_vacuum_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, rfe_vacuum_transformation_json):
        d = json.loads(
            rfe_vacuum_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = HybridTopProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(16.94, abs=0.5)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(0.16, abs=0.2)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)
