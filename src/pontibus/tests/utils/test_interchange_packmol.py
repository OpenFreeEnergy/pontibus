# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
import logging
from openmm import NonbondedForce, CustomNonbondedForce
from openff.toolkit import Molecule
from openff.units import unit
from openff.units.openmm import to_openmm, from_openmm
from gufe import SmallMoleculeComponent, SolventComponent
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.components.extended_solvent_component import (
    ExtendedSolventComponent,
)
from pontibus.utils.system_creation import (
    interchange_packmol_creation,
    _set_offmol_resname,
    _get_offmol_resname,
)
from pontibus.utils.molecules import WATER
from numpy.testing import assert_allclose, assert_equal


@pytest.fixture()
def smc_components_benzene(benzene_modifications):
    benzene_off = benzene_modifications["benzene"].to_openff()
    benzene_off.assign_partial_charges(partial_charge_method="gasteiger")
    return {benzene_modifications["benzene"]: benzene_off}


@pytest.fixture()
def methanol():
    m = Molecule.from_smiles("CO")
    m.generate_conformers()
    m.assign_partial_charges(partial_charge_method="gasteiger")
    return m


def test_protein_component_fail(smc_components_benzene, T4_protein_component):
    errmsg = "ProteinComponents is not currently supported"
    with pytest.raises(ValueError, match=errmsg):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene,
            protein_component=T4_protein_component,
            solvent_component=None,
            solvent_offmol=None,
        )


def test_get_and_set_offmol_resname(CN_molecule, caplog):
    CN_off = CN_molecule.to_openff()

    # No residue name to begin with
    assert _get_offmol_resname(CN_off) is None

    # Boop the floof
    _set_offmol_resname(CN_off, "BOOP")

    # Does the floof accept the boop?
    assert "BOOP" == _get_offmol_resname(CN_off)

    # Oh no, one of the atoms didn't like the boop!
    atom3 = list(CN_off.atoms)[2]
    atom3.metadata["residue_name"] = "NOBOOP"

    with caplog.at_level(logging.WARNING):
        assert _get_offmol_resname(CN_off) is None
    assert "Inconsistent residue name" in caplog.text


@pytest.mark.parametrize(
    "neutralize, ion_conc",
    [
        [True, 0.0 * unit.molar],
        [False, 0.1 * unit.molar],
        [True, 0.1 * unit.molar],
    ],
)
def test_wrong_solventcomp_settings(neutralize, ion_conc, smc_components_benzene):
    with pytest.raises(ValueError, match="Adding counterions"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=neutralize,
                ion_concentration=ion_conc,
            ),
            solvent_offmol=None,
        )


def test_solv_but_no_solv_offmol(
    smc_components_benzene,
):
    with pytest.raises(ValueError, match="A solvent offmol"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=None,
        )


def test_solv_mismatch(
    smc_components_benzene,
    methanol,
):
    assert ExtendedSolventComponent().smiles == "[H][O][H]"
    with pytest.raises(ValueError, match="does not match"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=methanol,
        )


def test_vacuum(smc_components_benzene):
    interchange, comp_resids = interchange_packmol_creation(
        ffsettings=InterchangeFFSettings(),
        solvation_settings=PackmolSolvationSettings(),
        smc_components=smc_components_benzene,
        protein_component=None,
        solvent_component=None,
        solvent_offmol=None,
    )

    assert len(comp_resids) == 1
    assert list(smc_components_benzene)[0] in comp_resids

    # Get the topology out
    omm_topology = interchange.to_openmm_topology()
    residues = list(omm_topology.residues())
    assert len(residues) == 1
    assert len(list(omm_topology.atoms())) == 12
    assert residues[0].name == "AAA"

    # Get the openmm system out..
    omm_system = interchange.to_openmm_system()

    nonbond = [f for f in omm_system.getForces() if isinstance(f, NonbondedForce)]

    # One nonbonded force
    assert len(nonbond) == 1

    # Gas phase should be nonbonded
    assert nonbond[0].getNonbondedMethod() == 0

    bond = [f for f in omm_system.getForces() if not isinstance(f, NonbondedForce)]

    # 3 bonded forces
    assert len(bond) == 3


def test_solvate_opc(smc_components_benzene):
    solvent_offmol = Molecule.from_smiles('O')
    interchange, comp_resids = interchange_packmol_creation(
        ffsettings=InterchangeFFSettings(
            forcefields=["openff-2.0.0.offxml", 'opc.offxml'],
        ),
        solvation_settings=PackmolSolvationSettings(),
        smc_components=smc_components_benzene,
        protein_component=None,
        solvent_component=SolventComponent(
            smiles='O', neutralize=False,
            ion_concentration=0 * unit.molar,
        ),
        solvent_offmol=solvent_offmol,
    )

    topology = interchange.to_openmm_topology()
    system = interchange.to_openmm_system()
    nonbonded_force = [
        f for f in system.getForces()
        if isinstance(f, NonbondedForce)
    ][0]
                       

    num_waters = topology.getNumResidues() - 1

    # Check particles
    # 12 benzene atoms + num_waters * 3 water atoms
    standard_particles = 12 + (num_waters * 3)

    # check some particles aren't virtual sites
    for i in range(standard_particles):
        assert not system.isVirtualSite(i)

    # check the water nonbonded values
    water_nonbonded_params = {
        1: [0 * unit.elementary_charge, 1.777167268 * 2 * unit.angstrom, 0.212800813 * unit.kilocalorie_per_mole]
        2: [0.679142 * unit.elementary_charge, 
        3: [
        4: [-0.679142 * 2 * unit.elementary_charge, 0.1 * unit.nanometer, 0 * unit.kilocalorie_per_mole],

    }
    for i in range(standard_particles - 12):
        c, s, e = nonbonded_force.getParticleParameters(i + 12)
        if i % 1

    for i in range(standard_particles, system.getNumParticles()):
        assert system.isVirtualSite(i)
        c, s, e = nonbonded_force.getParticleParameters(i)
        assert from_openmm(c) == -0.679142 * 2 * unit.elementary_charge
        assert from_openmm(e) == 0 * unit.kilocalorie_per_mole
        assert from_openmm(s) == 1 * unit.angstrom


    # Check that the openmm topology has the right indices
    # TODO: at least raise an issue - this is SUPER SUPER problematic
    ## honestly this might be the reason I will refuse to implement vsites
    ## eps = [at.id for at in topology.atoms()
    ##        if at.name == 'EP']
    ## assert_equal(
    ##     eps,
    ##     [i for i in range(standard_particles, system.getNumParticles())]
    ## )


#def test_library_charges_opc3(smc_components_benzene):
#    ...
#
#
#def test_setcharge_water_solvent(smc_components_benzene):
#    ...
#
#
#def test_setcharge_coc_solvent(smc_components_benzene):
#    ...
#

def test_noncharge_nolibrarycharges(smc_components_benzene):
    solvent_offmol = Molecule.from_smiles('COC')

    with pytest.raises(ValueError, match="No library charges"):
        _, _ = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml",]
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene,
            protein_component=None,
            solvent_component=SolventComponent(
                smiles='COC', neutralize=False,
                ion_concentration=0 * unit.molar,
            ),
            solvent_offmol=solvent_offmol,
        )


#def test_precharged_named(smc_components_benzene):
#    ...
#
#
#def test_precharged_unamed(smc_components_benzene):
#    ...
#
#
#def test_inconsistent_solvent_name(smc_components_benzene):
#    ...
#
#
#def test_duplicate_named_smcs(smc_components_benzene):
#    ...
#
#
#def test_box_setting_cube(smc_components_benzene):
#    ...
#
#
#def test_box_setting_dodecahedron(smc_components_benzene):
#    ...



"""
4. Named solvent
5. Unamed solvent
  - Check we get the new residue names
  - Check we get warned about renaming
6. Named solvent with inconsistent name
7. Duplicate named smcs
10. Cube
11. Dodecahedron
12. Check we get the right residues
13. Check we get the right number of atoms
  - with a solvent w/ virtual sites
  - check omm topology indices match virtual sites
"""
