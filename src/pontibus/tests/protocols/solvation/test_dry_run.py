# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from unittest import mock

import mdtraj as mdt
import pytest
from gufe import ChemicalSystem
from openff.units import unit

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation import ASFEProtocol, ASFESolventUnit, ASFEVacuumUnit


@pytest.mark.parametrize("method", ["repex", "sams", "independent", "InDePeNdENT"])
def test_dry_run_vacuum_benzene(benzene_modifications, method, tmpdir):
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.vacuum_simulation_settings.sampler_method = method

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": ExtendedSolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": ExtendedSolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units if isinstance(u, ASFEVacuumUnit)]
    sol_unit = [u for u in prot_units if isinstance(u, ASFESolventUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        vac_sampler = vac_unit[0].run(dry=True)["debug"]["sampler"]
        assert not vac_sampler.is_periodic


@pytest.mark.parametrize('experimental', [True, False])
def test_dry_run_solv_benzene(experimental, benzene_modifications, tmpdir):
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.solvent_output_settings.output_indices = "resname AAA"
    s.alchemical_settings.experimental = experimental

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": ExtendedSolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": ExtendedSolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units if isinstance(u, ASFEVacuumUnit)]
    sol_unit = [u for u in prot_units if isinstance(u, ASFESolventUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)["debug"]["sampler"]
        assert sol_sampler.is_periodic

        pdb = mdt.load_pdb("hybrid_system.pdb")
        assert pdb.n_atoms == 12


def test_dry_run_solv_benzene_opc(benzene_modifications, tmpdir):
    # TODO: validation tests
    # - hmass
    # - timestep
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.vacuum_forcefield_settings.forcefields = ["openff-2.0.0.offxml", "opc.offxml"]
    s.vacuum_forcefield_settings.hydrogen_mass = 1.0
    s.solvent_forcefield_settings.forcefields = ["openff-2.0.0.offxml", "opc.offxml"]
    s.solvent_forcefield_settings.hydrogen_mass = 1.007947
    s.integrator_settings.reassign_velocities = True
    s.integrator_settings.timestep = 2 * unit.femtosecond

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": ExtendedSolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": ExtendedSolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_unit = [u for u in prot_units if isinstance(u, ASFESolventUnit)]

    with tmpdir.as_cwd():
        sol_sampler = sol_unit[0].run(dry=True)["debug"]["sampler"]
        assert sol_sampler.is_periodic

        pdb = mdt.load_pdb("hybrid_system.pdb")
        assert pdb.n_atoms == 12


def test_confgen_fail_AFE(benzene_modifications, tmpdir):
    # check system parametrisation works even if confgen fails
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": ExtendedSolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": ExtendedSolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    vac_unit = [u for u in prot_units if isinstance(u, ASFEVacuumUnit)]

    with tmpdir.as_cwd():
        with mock.patch("rdkit.Chem.AllChem.EmbedMultipleConfs", return_value=0):
            vac_sampler = vac_unit[0].run(dry=True)["debug"]["sampler"]

            assert vac_sampler


"""
Different solvents
  - Implicit
  - Explicit
  - Virtual sites
Partial Charges
  - Implicit
  - Explicit
"""
