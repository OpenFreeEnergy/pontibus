# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import mdtraj as mdt
import numpy as np
import openfe
import pytest
from openff.units import unit
from openff.units.openmm import ensure_quantity
from openmm import NonbondedForce, MonteCarloBarostat
from openmm import unit as omm_unit
from openmmtools.multistate import MultiStateSampler
from rdkit import Chem

from pontibus.protocols.relative import HybridTopProtocol


def test_create_default_settings():
    settings = HybridTopProtocol.default_settings()

    assert settings


def test_create_default_protocol():
    protocol = HybridTopProtocol(settings=HybridTopProtocol.default_settings())

    assert protocol


def test_serialize_protocol():
    protocol = HybridTopProtocol(
        settings=HybridTopProtocol.default_settings(),
    )

    ser = protocol.to_dict()

    ret = HybridTopProtocol.from_dict(ser)

    assert protocol == ret


@pytest.mark.parametrize("method", ["repex", "sams", "independent", "InDePeNdENT"])
def test_dry_run_default_vacuum(
    benzene_vacuum_system, toluene_vacuum_system, benzene_to_toluene_mapping, method, tmpdir
):
    vac_settings = HybridTopProtocol.default_settings()
    vac_settings.forcefield_settings.nonbonded_method = "nocutoff"
    vac_settings.simulation_settings.sampler_method = method
    vac_settings.protocol_repeats = 1

    protocol = HybridTopProtocol(
        settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=toluene_vacuum_system,
        mapping=benzene_to_toluene_mapping,
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)["debug"]["sampler"]
        assert isinstance(sampler, MultiStateSampler)
        assert not sampler.is_periodic
        assert sampler._thermodynamic_states[0].barostat is None

        # Check hybrid OMM and MDTtraj Topologies
        htf = sampler._hybrid_factory
        # 16 atoms:
        # 11 common atoms, 1 extra hydrogen in benzene, 4 extra in toluene
        # 12 bonds in benzene + 4 extra toluene bonds
        assert len(list(htf.hybrid_topology.atoms)) == 16
        assert len(list(htf.omm_hybrid_topology.atoms())) == 16
        assert len(list(htf.hybrid_topology.bonds)) == 16
        assert len(list(htf.omm_hybrid_topology.bonds())) == 16

        # smoke test - can convert back the mdtraj topology
        ret_top = mdt.Topology.to_openmm(htf.hybrid_topology)
        assert len(list(ret_top.atoms())) == 16
        assert len(list(ret_top.bonds())) == 16

        # check that our PDB has the right number of atoms
        pdb = mdt.load_pdb("hybrid_system.pdb")
        assert pdb.n_atoms == 16


BENZ = """\
benzene
  PyMOL2.5          3D                             0

 12 12  0  0  0  0  0  0  0  0999 V2000
    1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7022    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5079   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2540    2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2540    2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5079   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2540   -2.1719    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2540   -2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  1  6  1  0  0  0  0
  1  7  1  0  0  0  0
  2  3  1  0  0  0  0
  2  8  1  0  0  0  0
  3  4  2  0  0  0  0
  3  9  1  0  0  0  0
  4  5  1  0  0  0  0
  4 10  1  0  0  0  0
  5  6  2  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
$$$$
"""


PYRIDINE = """\
pyridine
  PyMOL2.5          3D                             0

 11 11  0  0  0  0  0  0  0  0999 V2000
    1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4940   -0.0325    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2473   -2.1604    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2473   -2.1604    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4945   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2753    2.1437    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7525    1.3034    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  1 11  2  0  0  0  0
  2  3  2  0  0  0  0
  2 10  1  0  0  0  0
  3  4  1  0  0  0  0
  3  9  1  0  0  0  0
  4  5  2  0  0  0  0
  4  8  1  0  0  0  0
  5  7  1  0  0  0  0
  2 11  1  0  0  0  0
M  END
$$$$
"""


def test_dry_core_element_change(tmpdir):
    benz = openfe.SmallMoleculeComponent(Chem.MolFromMolBlock(BENZ, removeHs=False))
    pyr = openfe.SmallMoleculeComponent(Chem.MolFromMolBlock(PYRIDINE, removeHs=False))

    mapping = openfe.LigandAtomMapping(
        benz, pyr, {0: 0, 1: 10, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 9, 9: 8, 10: 7, 11: 6}
    )

    settings = HybridTopProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"

    protocol = HybridTopProtocol(
        settings=settings,
    )

    dag = protocol.create(
        stateA=openfe.ChemicalSystem(
            {
                "ligand": benz,
            }
        ),
        stateB=openfe.ChemicalSystem(
            {
                "ligand": pyr,
            }
        ),
        mapping=mapping,
    )

    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)["debug"]["sampler"]
        system = sampler._hybrid_factory.hybrid_system
        assert system.getNumParticles() == 12
        # Average mass between nitrogen and carbon
        assert pytest.approx(system.getParticleMass(1)._value) == 12.0008030

        # Get out the CustomNonbondedForce
        cnf = [f for f in system.getForces() if f.__class__.__name__ == "CustomNonbondedForce"][0]
        # there should be no new unique atoms
        assert cnf.getInteractionGroupParameters(6) == [(), ()]
        # there should be one old unique atom (spare hydrogen from the benzene)
        assert cnf.getInteractionGroupParameters(7) == [(7,), (7,)]


@pytest.mark.parametrize("method", ["repex", "sams", "independent"])
def test_dry_run_ligand(benzene_system, toluene_system, benzene_to_toluene_mapping, method, tmpdir):
    # this might be a bit time consuming
    settings = HybridTopProtocol.default_settings()
    settings.simulation_settings.sampler_method = method
    settings.protocol_repeats = 1
    settings.output_settings.output_indices = "resname UNK"

    protocol = HybridTopProtocol(
        settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=toluene_system,
        mapping=benzene_to_toluene_mapping,
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)["debug"]["sampler"]
        assert isinstance(sampler, MultiStateSampler)
        assert sampler.is_periodic
        assert isinstance(sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)
        assert sampler._thermodynamic_states[1].pressure == 1 * omm_unit.bar

        # Check we have the right number of atoms in the PDB
        pdb = mdt.load_pdb("hybrid_system.pdb")
        assert pdb.n_atoms == 16


def test_dry_run_user_charges(benzene_modifications, tmpdir):
    """
    Create a hybrid system with a set of fictitious user supplied charges
    and ensure that they are properly passed through to the constructed
    hybrid topology.
    """
    vac_settings = HybridTopProtocol.default_settings()
    vac_settings.forcefield_settings.nonbonded_method = "nocutoff"
    vac_settings.protocol_repeats = 1

    protocol = HybridTopProtocol(
        settings=vac_settings,
    )

    def assign_fictitious_charges(offmol):
        """
        Get a random array of fake partial charges (ints because why not)
        that sums up to 0. Note that OpenFF will complain if you try to
        create a molecule that has a total charge that is different from
        the expected formal charge, hence we enforce a zero charge here.
        """
        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
        rand_arr[-1] = -sum(rand_arr[:-1])
        return rand_arr * unit.elementary_charge

    def check_propchgs(smc, charge_array):
        """
        Check that the partial charges we assigned to our offmol from which
        the smc was constructed are present and the right ones.
        """
        prop_chgs = smc.to_dict()["molprops"]["atom.dprop.PartialCharge"]
        prop_chgs = np.array(prop_chgs.split(), dtype=float)
        np.testing.assert_allclose(prop_chgs, charge_array.m)

    # Create new smc with overriden charges
    benzene_offmol = benzene_modifications["benzene"].to_openff()
    toluene_offmol = benzene_modifications["toluene"].to_openff()
    benzene_rand_chg = assign_fictitious_charges(benzene_offmol)
    toluene_rand_chg = assign_fictitious_charges(toluene_offmol)
    benzene_offmol.partial_charges = benzene_rand_chg
    toluene_offmol.partial_charges = toluene_rand_chg
    benzene_smc = openfe.SmallMoleculeComponent.from_openff(benzene_offmol)
    toluene_smc = openfe.SmallMoleculeComponent.from_openff(toluene_offmol)

    # Check that the new smcs have the new overriden charges
    check_propchgs(benzene_smc, benzene_rand_chg)
    check_propchgs(toluene_smc, toluene_rand_chg)

    # Create new mapping
    mapper = openfe.setup.LomapAtomMapper(element_change=False)
    mapping = next(mapper.suggest_mappings(benzene_smc, toluene_smc))

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=openfe.ChemicalSystem(
            {
                "l": benzene_smc,
            }
        ),
        stateB=openfe.ChemicalSystem(
            {
                "l": toluene_smc,
            }
        ),
        mapping=mapping,
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)["debug"]["sampler"]
        htf = sampler._factory
        hybrid_system = htf.hybrid_system

        # get the standard nonbonded force
        nonbond = [f for f in hybrid_system.getForces() if isinstance(f, NonbondedForce)]
        assert len(nonbond) == 1

        # get the particle parameter offsets
        c_offsets = {}
        for i in range(nonbond[0].getNumParticleParameterOffsets()):
            offset = nonbond[0].getParticleParameterOffset(i)
            c_offsets[offset[1]] = ensure_quantity(offset[2], "openff")

        # Here is a bit of exposition on what we're doing
        # HTF creates two sets of nonbonded forces, a standard one (for the
        # PME) and a custom one (for sterics).
        # Here we specifically check charges, so we only concentrate on the
        # standard NonbondedForce.
        # The way the NonbondedForce is constructed is as follows:
        # - unique old atoms:
        #  * The particle charge is set to the input molA particle charge
        #  * The chargeScale offset is set to the negative value of the molA
        #    particle charge (such that by scaling you effectively zero out
        #    the charge.
        # - unique new atoms:
        #  * The particle charge is set to zero (doesn't exist in the starting
        #    end state).
        #  * The chargeScale offset is set to the value of the molB particle
        #    charge (such that by scaling you effectively go from 0 to molB
        #    charge).
        # - core atoms:
        #  * The particle charge is set to the input molA particle charge
        #    (i.e. we start from a system that has molA charges).
        #  * The particle charge offset is set to the difference between
        #    the molB particle charge and the molA particle charge (i.e.
        #    we scale by that difference to get to the value of the molB
        #    particle charge).
        for i in range(hybrid_system.getNumParticles()):
            c, s, e = nonbond[0].getParticleParameters(i)
            # get the particle charge (c)
            c = ensure_quantity(c, "openff")
            # particle charge (c) is equal to molA particle charge
            # offset (c_offsets) is equal to -(molA particle charge)
            if i in htf._atom_classes["unique_old_atoms"]:
                idx = htf._hybrid_to_old_map[i]
                np.testing.assert_allclose(c, benzene_rand_chg[idx])
                np.testing.assert_allclose(c_offsets[i], -benzene_rand_chg[idx])
            # particle charge (c) is equal to 0
            # offset (c_offsets) is equal to molB particle charge
            elif i in htf._atom_classes["unique_new_atoms"]:
                idx = htf._hybrid_to_new_map[i]
                np.testing.assert_allclose(c, 0 * unit.elementary_charge)
                np.testing.assert_allclose(c_offsets[i], toluene_rand_chg[idx])
            # particle charge (c) is equal to molA particle charge
            # offset (c_offsets) is equal to difference between molB and molA
            elif i in htf._atom_classes["core_atoms"]:
                old_i = htf._hybrid_to_old_map[i]
                new_i = htf._hybrid_to_new_map[i]
                c_exp = toluene_rand_chg[new_i] - benzene_rand_chg[old_i]
                np.testing.assert_allclose(c, benzene_rand_chg[old_i])
                np.testing.assert_allclose(c_offsets[i], c_exp)
