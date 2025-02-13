# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json

import gufe
import numpy as np
import openfe
import pytest
from openff.units import unit as offunit

from pontibus.protocols.solvation import ASFEProtocolResult


class TestWaterProtocolResult:
    @pytest.fixture()
    def protocolresult(self, afe_solv_water_transformation_json):
        d = json.loads(
            afe_solv_water_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, afe_solv_water_transformation_json):
        d = json.loads(
            afe_solv_water_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = ASFEProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(-2.47, abs=0.5)
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

        assert isinstance(inds, dict)
        assert isinstance(inds["solvent"], list)
        assert isinstance(inds["vacuum"], list)
        assert len(inds["solvent"]) == len(inds["vacuum"]) == 3
        for e, u in itertools.chain(inds["solvent"], inds["vacuum"]):
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_forwards_etc(self, key, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, dict)
        assert isinstance(far[key], list)
        far1 = far[key][0]
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

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_frwd_reverse_none_return(self, key, protocolresult):
        # fetch the first result of type key
        data = [i for i in protocolresult.data[key].values()][0][0]
        # set the output to None
        data.outputs["forward_and_reverse_energies"] = None

        # now fetch the analysis results and expect a warning
        wmsg = (
            "were found in the forward and reverse dictionaries "
            f"of the repeats of the {key}"
        )
        with pytest.warns(UserWarning, match=wmsg):
            protocolresult.get_forward_and_reverse_energy_analysis()

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_overlap_matrices(self, key, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, dict)
        assert isinstance(ovp[key], list)
        assert len(ovp[key]) == 3

        ovp1 = ovp[key][0]
        assert isinstance(ovp1["matrix"], np.ndarray)
        assert ovp1["matrix"].shape == (14, 14)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_get_replica_transition_statistics(self, key, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()

        assert isinstance(rpx, dict)
        assert isinstance(rpx[key], list)
        assert len(rpx[key]) == 3
        rpx1 = rpx[key][0]
        assert "eigenvalues" in rpx1
        assert "matrix" in rpx1
        assert rpx1["eigenvalues"].shape == (14,)
        assert rpx1["matrix"].shape == (14, 14)

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_equilibration_iterations(self, key, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, dict)
        assert isinstance(eq[key], list)
        assert len(eq[key]) == 3
        assert all(isinstance(v, float) for v in eq[key])

    @pytest.mark.parametrize("key", ["solvent", "vacuum"])
    def test_production_iterations(self, key, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, dict)
        assert isinstance(prod[key], list)
        assert len(prod[key]) == 3
        assert all(isinstance(v, float) for v in prod[key])

    def test_filenotfound_replica_states(self, protocolresult):
        errmsg = "File could not be found"

        with pytest.raises(ValueError, match=errmsg):
            protocolresult.get_replica_states()


class TestOctanolProtocolResult(TestWaterProtocolResult):
    @pytest.fixture()
    def protocolresult(self, afe_solv_octanol_transformation_json):
        d = json.loads(
            afe_solv_octanol_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, afe_solv_octanol_transformation_json):
        d = json.loads(
            afe_solv_octanol_transformation_json,
            cls=gufe.tokenization.JSON_HANDLER.decoder,
        )

        pr = ASFEProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(-4.83, abs=0.5)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est
        assert est.m == pytest.approx(0.47, abs=0.2)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)
