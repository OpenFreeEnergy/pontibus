# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from unittest import mock

import mdtraj as mdt
import numpy as np
import pytest
from gufe import ChemicalSystem, SmallMoleculeComponent
from openfe.protocols.openmm_utils.charge_generation import (
    assign_offmol_partial_charges,
)
from openfe.tests.protocols.openmm_ahfe.test_ahfe_protocol import (
    _assert_num_forces,
    _verify_alchemical_sterics_force_parameters,
)
from openff.toolkit import ForceField, Molecule
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
from openmmtools.multistate import MultiStateSampler

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation import (
    ASFEProtocol,
    ASFESolventSetupUnit,
    ASFESolventSimUnit,
    ASFEVacuumSetupUnit,
    ASFEVacuumSimUnit,
)


def _get_units(protocol_units, unit_type):
    """
    Helper method to extract setup units
    """
    return [pu for pu in protocol_units if isinstance(pu, unit_type)]


@pytest.fixture()
def dry_settings():
    settings = ASFEProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.vacuum_engine_settings.compute_platform = None
    settings.solvent_engine_settings.compute_platform = None
    return settings


@pytest.mark.parametrize("method", ["repex", "sams", "independent", "InDePeNdENT"])
def test_dry_run_vacuum_benzene(charged_benzene, dry_settings, method, tmpdir):
    dry_settings.vacuum_simulation_settings.sampler_method = method

    protocol = ASFEProtocol(
        settings=dry_settings,
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

    assert len(prot_units) == 6

    vac_setup_unit = _get_units(dag.protocol_units, ASFEVacuumSetupUnit)[0]
    vac_sim_unit = _get_units(dag.protocol_units, ASFEVacuumSimUnit)[0]

    with tmpdir.as_cwd():
        setup_results = vac_setup_unit.run(dry=True)
        sim_results = vac_sim_unit.run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            dry=True,
        )

        sampler = sim_results["sampler"]
        assert isinstance(sampler, MultiStateSampler)
        assert not sampler.is_periodic

        system = setup_results["alchem_system"]
        assert len(system.getForces()) == 12

        _assert_num_forces(system, NonbondedForce, 1)
        _assert_num_forces(system, CustomNonbondedForce, 4)
        _assert_num_forces(system, CustomBondForce, 4)
        _assert_num_forces(system, HarmonicBondForce, 1)
        _assert_num_forces(system, HarmonicAngleForce, 1)
        _assert_num_forces(system, PeriodicTorsionForce, 1)

        # Check the nonbonded force is NoCutoff
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.NoCutoff


@pytest.mark.parametrize("experimental", [True, False])
@pytest.mark.parametrize(
    "alpha, a, b, c, correction",
    [
        [0.2, 2, 2, 1, True],
        [0.35, 2.2, 1.5, 0, False],
    ],
)
def test_dry_run_solv_benzene(
    experimental, alpha, a, b, c, correction, charged_benzene, dry_settings, tmpdir
):
    dry_settings.solvent_output_settings.output_indices = "resname LIG"
    # Set a non-default barostat frequency to make sure it goes all the way
    dry_settings.integrator_settings.barostat_frequency = 125 * unit.timestep
    dry_settings.alchemical_settings.experimental = experimental
    dry_settings.alchemical_settings.softcore_alpha = alpha
    dry_settings.alchemical_settings.softcore_a = a
    dry_settings.alchemical_settings.softcore_b = b
    dry_settings.alchemical_settings.softcore_c = c
    dry_settings.alchemical_settings.disable_alchemical_dispersion_correction = correction

    protocol = ASFEProtocol(
        settings=dry_settings,
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

    assert len(prot_units) == 6

    sol_setup_unit = _get_units(dag.protocol_units, ASFESolventSetupUnit)[0]
    sol_sim_unit = _get_units(dag.protocol_units, ASFESolventSimUnit)[0]

    with tmpdir.as_cwd():
        setup_results = sol_setup_unit.run(dry=True)
        sim_results = sol_sim_unit.run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            dry=True,
        )

        sampler = sim_results["sampler"]
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic
        assert isinstance(sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)

        pdb = mdt.load_pdb("hybrid_system.pdb")
        assert pdb.n_atoms == 12

        system = setup_results["alchem_system"]
        assert len(system.getForces()) == 9

        _assert_num_forces(system, NonbondedForce, 1)
        _assert_num_forces(system, CustomNonbondedForce, 2)
        _assert_num_forces(system, CustomBondForce, 2)
        _assert_num_forces(system, HarmonicBondForce, 1)
        _assert_num_forces(system, HarmonicAngleForce, 1)
        _assert_num_forces(system, PeriodicTorsionForce, 1)
        _assert_num_forces(system, MonteCarloBarostat, 1)

        # Check the initial barostat made it all the way through
        for force in system.getForces():
            if isinstance(force, MonteCarloBarostat):
                assert force.getFrequency() == 125
                assert (
                    from_openmm(force.getDefaultPressure()) == dry_settings.thermo_settings.pressure
                )
                assert (
                    from_openmm(force.getDefaultTemperature())
                    == dry_settings.thermo_settings.temperature
                )

        # Check the nonbonded force is PME
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.PME

        # Check custom steric force contents
        stericsf = [
            f
            for f in system.getForces()
            if isinstance(f, CustomNonbondedForce) and "U_sterics" in f.getEnergyFunction()
        ]

        for force in stericsf:
            _verify_alchemical_sterics_force_parameters(
                force,
                long_range=not correction,
                alpha=alpha,
                a=a,
                b=b,
                c=c,
            )


def test_dry_run_benzene_in_benzene_user_charges(charged_benzene, dry_settings, tmpdir):
    """
    A basic user charges test - i.e. will it retain _some_ charges passed
    through.

    TODO: something a bit more intensive.
    """
    dry_settings.solvent_output_settings.output_indices = "resname AAA"
    dry_settings.solvation_settings.assign_solvent_charges = True

    protocol = ASFEProtocol(
        settings=dry_settings,
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
    sol_setup_unit = _get_units(dag.protocol_units, ASFESolventSetupUnit)[0]

    with tmpdir.as_cwd():
        setup_results = sol_setup_unit.run(dry=True)
        system = setup_results["alchem_system"]

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


def test_dry_run_solv_benzene_opc(charged_benzene, dry_settings, tmpdir):
    # TODO: validation tests
    # - hmass
    # - timestep
    dry_settings.vacuum_forcefield_settings.forcefields = ["openff-2.0.0.offxml", "opc.offxml"]
    dry_settings.vacuum_forcefield_settings.hydrogen_mass = 1.0
    dry_settings.solvent_forcefield_settings.forcefields = ["openff-2.0.0.offxml", "opc.offxml"]
    dry_settings.solvent_forcefield_settings.hydrogen_mass = 1.007947
    dry_settings.integrator_settings.reassign_velocities = True
    dry_settings.integrator_settings.timestep = 2 * unit.femtosecond

    protocol = ASFEProtocol(
        settings=dry_settings,
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
    sol_setup_unit = _get_units(dag.protocol_units, ASFESolventSetupUnit)[0]
    sol_sim_unit = _get_units(dag.protocol_units, ASFESolventSimUnit)[0]

    with tmpdir.as_cwd():
        setup_results = sol_setup_unit.run(dry=True)
        sim_results = sol_sim_unit.run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            dry=True,
        )
        sampler = sim_results["sampler"]
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic
        assert isinstance(sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)

        pdb = mdt.load_pdb("hybrid_system.pdb")
        assert pdb.n_atoms == 12


@pytest.fixture(scope="module")
def bromopyridine_ff_charged(bromopyridine_sdf, vsite_offxml):
    """
    3-bromopyridine with partial charges assigned from the vsite
    force field (i.e. base charges prior to vsite charge increments).
    """
    offmol = Molecule.from_file(bromopyridine_sdf)
    with pytest.warns(UserWarning, match="VirtualSiteHandler"):
        assign_offmol_partial_charges(
            offmol=offmol,
            overwrite=False,
            method="forcefield",
            toolkit_backend="ambertools",
            generate_n_conformers=None,
            nagl_model=None,
            forcefields=[vsite_offxml],
        )
    return SmallMoleculeComponent.from_openff(offmol)


@pytest.fixture(scope="module")
def bromopyridine_reference_charges(bromopyridine_sdf, vsite_offxml):
    """
    Reference partial charges (atoms followed by virtual sites) from a
    vacuum system created natively with Interchange.
    """
    offmol = Molecule.from_file(bromopyridine_sdf)
    assert offmol.partial_charges is None

    interchange = ForceField(vsite_offxml).create_interchange(offmol.to_topology())
    system = interchange.to_openmm_system()
    nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    charges = np.array(
        [
            ensure_quantity(nonbond.getParticleParameters(i)[0], "openff").m
            for i in range(system.getNumParticles())
        ]
    )
    is_vsite = [system.isVirtualSite(i) for i in range(system.getNumParticles())]
    return charges, is_vsite


def _get_nonbonded_charges(system, indices):
    nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    assert len(nonbond) == 1
    return np.array(
        [ensure_quantity(nonbond[0].getParticleParameters(i)[0], "openff").m for i in indices]
    )


@pytest.mark.parametrize("unit_type", [ASFEVacuumSetupUnit, ASFESolventSetupUnit])
def test_dry_run_vsite_ff_charges(
    unit_type,
    bromopyridine_ff_charged,
    bromopyridine_reference_charges,
    vsite_offxml,
    dry_settings,
    tmpdir,
):
    """
    Check that partial charges assigned using the openfe `forcefield`
    charge method on a molecule with multiple virtual sites (a halogen
    BondCharge and a pyridine DivalentLonePair) yield the same System
    charges as a native Interchange parameterization.
    """
    ref_charges, ref_is_vsite = bromopyridine_reference_charges
    n_atoms = bromopyridine_ff_charged.to_openff().n_atoms

    # sanity check the reference: 11 atoms + 2 vsites
    assert n_atoms == 11
    assert len(ref_charges) == 13
    assert sum(ref_is_vsite) == 2
    assert ref_is_vsite == [False] * 11 + [True] * 2

    for ffsettings in [
        dry_settings.vacuum_forcefield_settings,
        dry_settings.solvent_forcefield_settings,
    ]:
        ffsettings.forcefields = [vsite_offxml]
    # vsites need velocity reassignment & the experimental factory
    dry_settings.integrator_settings.reassign_velocities = True
    dry_settings.alchemical_settings.experimental = True

    protocol = ASFEProtocol(settings=dry_settings)

    stateA = ChemicalSystem(
        {
            "solute": bromopyridine_ff_charged,
            "solvent": ExtendedSolventComponent(),
        }
    )
    stateB = ChemicalSystem({"solvent": ExtendedSolventComponent()})

    dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None)
    setup_unit = _get_units(dag.protocol_units, unit_type)[0]

    with tmpdir.as_cwd():
        setup_results = setup_unit.run(dry=True)

    # Solute particles: atoms followed by its virtual sites, in the same
    # order as the reference (in solvent the vsites come after the water)
    alchem_indices = setup_results["alchem_indices"]
    assert len(alchem_indices) == 13
    assert alchem_indices[:n_atoms] == list(range(n_atoms))

    # Standard (non-alchemical) system
    system = setup_results["standard_system"]
    # Exactly 2 virtual sites in the whole system, both on the solute
    vsites = [i for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
    assert len(vsites) == 2
    assert set(vsites) <= set(alchem_indices)
    assert [system.isVirtualSite(i) for i in alchem_indices] == ref_is_vsite
    system_charges = _get_nonbonded_charges(system, alchem_indices)
    assert system_charges == pytest.approx(ref_charges, abs=1e-6)
    assert system_charges.sum() == pytest.approx(0, abs=1e-6)

    # Check that the vsite charge increments were actually applied
    # on top of the input charges.
    input_charges = bromopyridine_ff_charged.to_openff().partial_charges.m
    assert np.all(system_charges[n_atoms:] != 0)
    assert not np.allclose(system_charges[:n_atoms], input_charges)

    # Alchemical system
    alchem_system = setup_results["alchem_system"]
    assert alchem_system.getNumParticles() == system.getNumParticles()
    assert [alchem_system.isVirtualSite(i) for i in alchem_indices] == ref_is_vsite

    # Solute charges are zeroed in the NonbondedForce...
    assert _get_nonbonded_charges(alchem_system, alchem_indices) == pytest.approx(0)

    # ...and held elsewhere
    if unit_type == ASFEVacuumSetupUnit:
        # In vacuum electrostatics live in CustomNonbondedForces
        elecf = [
            f
            for f in alchem_system.getForces()
            if isinstance(f, CustomNonbondedForce) and "U_electrostatics" in f.getEnergyFunction()
        ]
        assert len(elecf) == 2
        for force in elecf:
            charge_idx = [
                force.getPerParticleParameterName(i)
                for i in range(force.getNumPerParticleParameters())
            ].index("charge")
            charges = np.array([force.getParticleParameters(i)[charge_idx] for i in alchem_indices])
            assert charges == pytest.approx(ref_charges, abs=1e-6)
    else:
        # In solvent the charges are lambda_electrostatics offsets
        nonbond = [f for f in alchem_system.getForces() if isinstance(f, NonbondedForce)][0]
        offsets = {}
        for i in range(nonbond.getNumParticleParameterOffsets()):
            param, idx, chg, _, _ = nonbond.getParticleParameterOffset(i)
            assert param == "lambda_electrostatics"
            offsets[idx] = chg
        assert set(offsets) == set(alchem_indices)
        offset_charges = np.array([offsets[i] for i in alchem_indices])
        assert offset_charges == pytest.approx(ref_charges, abs=1e-6)

        # Non-alchemical (solvent) charges are untouched
        other = [i for i in range(system.getNumParticles()) if i not in set(alchem_indices)]
        assert _get_nonbonded_charges(alchem_system, other) == pytest.approx(
            _get_nonbonded_charges(system, other)
        )


def test_confgen_fail_AFE(benzene_modifications, dry_settings, tmpdir):
    # check system parametrisation works even if confgen fails
    protocol = ASFEProtocol(
        settings=dry_settings,
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
    vac_setup_unit = _get_units(dag.protocol_units, ASFEVacuumSetupUnit)[0]

    with tmpdir.as_cwd():
        with mock.patch("rdkit.Chem.AllChem.EmbedMultipleConfs", return_value=0):
            result = vac_setup_unit.run(dry=True)
            assert result


"""
Different solvents
  - Implicitly defined (i.e. smiles only)
  - Virtual sites (checking you have the right vsites and parameters)
Partial Charges
  - Implicit (calculated on the fly)
  - Explicit (something a bit more comprehensive than the benzene in benzene test)
"""
