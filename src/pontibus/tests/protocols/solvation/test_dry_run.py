# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from unittest import mock

import mdtraj as mdt
import pytest
from gufe import ChemicalSystem
from openff.units import unit
from openff.units.openmm import ensure_quantity, from_openmm
from openmm import (
    CustomBondForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
)

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation import ASFEProtocol, ASFESolventUnit, ASFEVacuumUnit


@pytest.mark.parametrize("method", ["repex", "sams", "independent", "InDePeNdENT"])
def test_dry_run_vacuum_benzene(charged_benzene, method, tmpdir):
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.vacuum_simulation_settings.sampler_method = method

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene,
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

        system = vac_sampler._thermodynamic_states[0].get_system(remove_thermostat=True)
        print(system.getForces())
        assert len(system.getForces()) == 12

        def assert_force_num(system, forcetype, number):
            forces = [f for f in system.getForces() if isinstance(f, forcetype)]
            assert len(forces) == number

        assert_force_num(system, NonbondedForce, 1)
        assert_force_num(system, CustomNonbondedForce, 4)
        assert_force_num(system, CustomBondForce, 4)
        assert_force_num(system, HarmonicBondForce, 1)
        assert_force_num(system, HarmonicAngleForce, 1)
        assert_force_num(system, PeriodicTorsionForce, 1)

        # Check the nonbonded force is NoCutoff
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.NoCutoff


@pytest.mark.parametrize("experimental", [True, False])
def test_dry_run_solv_benzene(experimental, charged_benzene, tmpdir):
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.solvent_output_settings.output_indices = "resname AAA"
    # Set a random barostat frequency to make sure it goes all the way
    s.integrator_settings.barostat_frequency = 125
    s.alchemical_settings.experimental = experimental

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene,
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

        system = sol_sampler._thermodynamic_states[0].get_system(remove_thermostat=True)
        assert len(system.getForces()) == 9

        def assert_force_num(system, forcetype, number):
            forces = [f for f in system.getForces() if isinstance(f, forcetype)]
            assert len(forces) == number

        assert_force_num(system, NonbondedForce, 1)
        assert_force_num(system, CustomNonbondedForce, 2)
        assert_force_num(system, CustomBondForce, 2)
        assert_force_num(system, HarmonicBondForce, 1)
        assert_force_num(system, HarmonicAngleForce, 1)
        assert_force_num(system, PeriodicTorsionForce, 1)
        assert_force_num(system, MonteCarloBarostat, 1)

        # Check the initial barostat made it all the way through
        for force in system.getForces():
            if isinstance(force, MonteCarloBarostat):
                assert force.getFrequency() == 125
                assert from_openmm(force.getDefaultPressure()) == s.thermo_settings.pressure
                assert from_openmm(force.getDefaultTemperature()) == s.thermo_settings.temperature

        # Check the nonbonded force is PME
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.PME


def test_dry_run_benzene_in_benzene_user_charges(charged_benzene, tmpdir):
    """
    A basic user charges test - i.e. will it retain _some_ charges passed
    through.

    TODO: something a bit more intensive.
    """
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.solvent_output_settings.output_indices = "resname AAA"
    s.solvation_settings.assign_solvent_charges = True

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene,
            "solvent": ExtendedSolventComponent(solvent_molecule=charged_benzene),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": ExtendedSolventComponent(solvent_molecule=charged_benzene),
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
        sampler = sol_unit[0].run(dry=True)["debug"]["sampler"]

        system = sampler._thermodynamic_states[0].system

        # Should be benzenes all the way down
        assert system.getNumParticles() % 12 == 0

        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]

        assert len(nonbond) == 1

        # loop through the benzene atoms
        # partial charge is stored in the offset
        prop_chgs = charged_benzene.to_openff().partial_charges
        for i in range(12):
            offsets = nonbond[0].getParticleParameterOffset(i)
            c = ensure_quantity(offsets[2], "openff")

            assert pytest.approx(c) == prop_chgs[i]

        for i in range(12, system.getNumParticles()):
            param = nonbond[0].getParticleParameters(i)
            c = ensure_quantity(param[0], "openff")

            benzene_idx = i % 12
            assert pytest.approx(c) == prop_chgs[benzene_idx]


def test_dry_run_solv_benzene_opc(charged_benzene, tmpdir):
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
            "benzene": charged_benzene,
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
  - Implicitly defined (i.e. smiles only)
  - Virtual sites (checking you have the right vsites and parameters)
Partial Charges
  - Implicit (calculated on the fly)
  - Explicit (something a bit more comprehensive than the benzene in benzene test)
"""
