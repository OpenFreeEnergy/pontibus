# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
import pathlib
from openff.units import unit
from gufe.protocols import execute_DAG
import numpy as np
from numpy.testing import assert_allclose
from pontibus.protocols.relative import HybridTopProtocol



@pytest.mark.gpu
def test_vacuum(
    benzene_vacuum_system,
    toluene_vacuum_system,
    benzene_to_toluene_mapping,
    tmpdir
):
    """
    Run a short MD simulation and make sure things didn't fail.
    """
    s = HybridTopProtocol.default_settings()
    s.simulation_settings.equilibration_length = 100 * unit.picosecond
    s.simulation_settings.production_length = 500 * unit.picosecond
    s.forcefield_settings.nonbonded_method = 'nocutoff'
    s.protocol_repeats = 1
    s.engine_settings.compute_platform = 'CUDA'

    p = HybridTopProtocol(s)

    dag = p.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping=benzene_to_toluene_mapping,
    )

    cwd = pathlib.Path(str(tmpdir))
    r = execute_DAG(
        dag,
        shared_basedir=cwd,
        scratch_basedir=cwd,
        keep_shared=True
    )

    assert r.ok()
    for pur in r.protocol_unit_results:
        unit_shared = tmpdir / f"shared_{pur.source_key}_attempt_0"
        assert unit_shared.exists()
        assert pathlib.Path(unit_shared).is_dir()

        # Check the checkpoint file exists
        checkpoint = pur.outputs['last_checkpoint']
        assert checkpoint == "checkpoint.chk"
        assert (unit_shared / checkpoint).exists()

        # Check the nc simulation file exists
        nc = pur.outputs['nc']
        assert nc == unit_shared / "simulation.nc"
        assert nc.exists()

        # Check structural analysis contents
        # TODO: for now this is disabled due to issue #117
        #structural_analysis_file = unit_shared / "structural_analysis.npz"
        #assert (structural_analysis_file).exists()
        #assert pur.outputs['structural_analysis'] == structural_analysis_file

        #structural_data = np.load(pur.outputs['structural_analysis'])
        #structural_keys = [
        #    'protein_RMSD', 'ligand_RMSD', 'ligand_COM_drift',
        #    'protein_2D_RMSD', 'time_ps'
        #]
        #for key in structural_keys:
        #    assert key in structural_data.keys()

        ## 6 frames being written to file
        #assert_allclose(structural_data['time_ps'], [0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
        #assert structural_data['ligand_RMSD'].shape == (11, 6)
        #assert structural_data['ligand_COM_drift'].shape == (11, 6)
        ## No protein so should be empty
        #assert structural_data['protein_RMSD'].size == 0
        #assert structural_data['protein_2D_RMSD'].size == 0

    # Test results
    results = p.gather([r])
    estimate = results.get_estimate()
    assert estimate.m == pytest.approx(0.80, abs=0.2)
    uncert = results.get_uncertainty()
    assert uncert.m == pytest.approx(0.0)
    states = results.get_replica_states()
    assert len(states) == 1
    assert states[0].shape[1] == 11
