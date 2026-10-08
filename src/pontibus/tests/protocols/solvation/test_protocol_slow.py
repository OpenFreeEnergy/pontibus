# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib

import pytest
from gufe import ChemicalSystem
from gufe.protocols import execute_DAG
from openff.units import unit

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation import ASFEProtocol


@pytest.mark.gpu  # takes too long to be a slow test ~ 4 mins locally
def test_openmm_run_engine(charged_benzene, tmp_path):
    """
    A basic integration test that purely checks if the whole Protocol will run.
    """
    platform = "CUDA"

    # Run a really short calculation to check everything is going well
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.solvent_output_settings.output_indices = "resname LIG"
    s.vacuum_equil_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.vacuum_equil_simulation_settings.production_length = 0.1 * unit.picosecond
    s.vacuum_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.vacuum_simulation_settings.production_length = 0.1 * unit.picosecond
    s.solvent_equil_simulation_settings.equilibration_length_nvt = 0.1 * unit.picosecond
    s.solvent_equil_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.solvent_equil_simulation_settings.production_length = 0.1 * unit.picosecond
    s.solvent_simulation_settings.equilibration_length = 0.1 * unit.picosecond
    s.solvent_simulation_settings.production_length = 0.1 * unit.picosecond
    s.vacuum_engine_settings.compute_platform = platform
    s.solvent_engine_settings.compute_platform = platform
    s.vacuum_simulation_settings.time_per_iteration = 20 * unit.femtosecond
    s.solvent_simulation_settings.time_per_iteration = 20 * unit.femtosecond
    s.vacuum_output_settings.checkpoint_interval = 20 * unit.femtosecond
    s.solvent_output_settings.checkpoint_interval = 20 * unit.femtosecond
    # Structural analysis needs more than one frame
    s.solvent_output_settings.positions_write_frequency = 20 * unit.femtosecond

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene,
            "solvent": ExtendedSolventComponent(),
        }
    )

    stateB = ChemicalSystem({"solvent": ExtendedSolventComponent()})

    # Create DAG from protocol and run all the units
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )

    r = execute_DAG(dag, shared_basedir=tmp_path, scratch_basedir=tmp_path, keep_shared=True)

    assert r.ok()

    # Check outputs of solvent & vacuum results
    for phase in ["solvent", "vacuum"]:
        purs = [pur for pur in r.protocol_unit_results if pur.outputs["simtype"] == phase]

        # get the path to the simulation unit shared dict
        for pur in purs:
            if "Simulation" in pur.name:
                sim_shared = tmp_path / f"shared_{pur.source_key}_attempt_0"
                assert sim_shared.exists()
                assert pathlib.Path(sim_shared).is_dir()

        # check the analysis outputs
        for pur in purs:
            if "Analysis" not in pur.name:
                continue

            unit_shared = tmp_path / f"shared_{pur.source_key}_attempt_0"
            assert unit_shared.exists()
            assert pathlib.Path(unit_shared).is_dir()

            # Does the checkpoint file exist?
            checkpoint = pur.outputs["checkpoint"]
            assert checkpoint == sim_shared / f"{pur.outputs['simtype']}_checkpoint.nc"
            assert checkpoint.exists()

            # Does the trajectory file exist?
            nc = pur.outputs["trajectory"]
            assert nc == sim_shared / f"{pur.outputs['simtype']}.nc"
            assert nc.exists()

            # Check the structural analysis outputs
            assert "structural_analysis_error" not in pur.outputs
            structural_pngs = ["ligand_RMSD.png", "ligand_COM_drift.png", "protein_2D_RMSD.png"]
            if phase == "vacuum":
                # No structural analysis is done in vacuum
                assert "structural_analysis" not in pur.outputs
                for png in structural_pngs:
                    assert not (unit_shared / png).exists()
            else:
                npz = pur.outputs["structural_analysis"]
                assert npz == unit_shared / "structural_analysis.npz"
                assert npz.exists()
                # Only the ligand RMSD is analyzed in solvent
                assert (unit_shared / "ligand_RMSD.png").exists()
                assert not (unit_shared / "ligand_COM_drift.png").exists()
                assert not (unit_shared / "protein_2D_RMSD.png").exists()

    # Test results methods that need files present
    results = protocol.gather([r])
    states = results.get_replica_states()
    assert len(states.items()) == 2
    assert len(states["solvent"]) == 1
    assert states["solvent"][0].shape[1] == 14
    assert len(states["vacuum"]) == 1
    assert states["vacuum"][0].shape[1] == 5
