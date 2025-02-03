# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import logging

import pytest
from gufe import SmallMoleculeComponent, SolventComponent
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)

from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.molecules import WATER
from pontibus.utils.system_creation import (
    _check_and_deduplicate_charged_mols,
    _check_library_charges,
    _get_offmol_resname,
    _set_offmol_resname,
    interchange_packmol_creation,
)


@pytest.fixture(scope="module")
def smc_components_benzene_unnamed(benzene_modifications):
    benzene_off = benzene_modifications["benzene"].to_openff()
    benzene_off.assign_partial_charges(partial_charge_method="gasteiger")
    return {benzene_modifications["benzene"]: benzene_off}


@pytest.fixture(scope="module")
def smc_components_benzene_named(benzene_modifications):
    benzene_off = benzene_modifications["benzene"].to_openff()
    _set_offmol_resname(benzene_off, "BNZ")
    benzene_off.assign_partial_charges(partial_charge_method="gasteiger")
    return {benzene_modifications["benzene"]: benzene_off}


@pytest.fixture()
def methanol():
    m = Molecule.from_smiles("CO")
    m.generate_conformers()
    m.assign_partial_charges(partial_charge_method="gasteiger")
    return m


@pytest.fixture(scope="module")
def water_off():
    return WATER.to_openff()


@pytest.fixture(scope="module")
def water_off_named_charged():
    water = WATER.to_openff()
    _set_offmol_resname(water, "HOH")
    water.assign_partial_charges(partial_charge_method="gasteiger")
    return water


@pytest.fixture(scope="module")
def water_off_am1bcc():
    water = WATER.to_openff()
    water.assign_partial_charges(partial_charge_method="am1bcc")
    return water


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


def test_check_library_charges_pass(water_off):
    ff = ForceField("opc.offxml")
    _check_library_charges(ff, water_off)


def test_check_library_charges_fail(methanol):
    ff = ForceField("openff-2.0.0.offxml")
    with pytest.raises(ValueError, match="No library charges"):
        _check_library_charges(ff, methanol)


def test_check_charged_mols_pass(methanol):
    _check_and_deduplicate_charged_mols([methanol])


def test_check_deduplicate_charged_mols(smc_components_benzene_unnamed):
    """
    Base test case for deduplication. Same molecule, same partial charges,
    different conformer.
    """
    benzene1 = list(smc_components_benzene_unnamed.values())[0]
    benzene1.assign_partial_charges(partial_charge_method="gasteiger")
    benzene2 = Molecule.from_smiles("c1ccccc1")
    benzene2.generate_conformers(n_conformers=1)
    benzene2.assign_partial_charges(partial_charge_method="gasteiger")

    assert all(benzene1.partial_charges == benzene2.partial_charges)
    assert np.any(benzene1.conformers[0] != benzene2.conformers[0])
    assert benzene1.is_isomorphic_with(benzene2)

    uniques = _check_and_deduplicate_charged_mols([benzene1, benzene2])

    assert len(uniques) == 1
    assert uniques[0] == benzene1


def test_check_charged_mols_nocharge(water_off, methanol):
    with pytest.raises(ValueError, match="One or more"):
        _check_and_deduplicate_charged_mols([water_off, methanol])


def test_check_charged_mols(water_off_am1bcc, water_off_named_charged):
    with pytest.raises(ValueError, match="different charges"):
        _check_and_deduplicate_charged_mols([water_off_am1bcc, water_off_named_charged])


def test_protein_component_fail(smc_components_benzene_named, T4_protein_component):
    errmsg = "ProteinComponents is not currently supported"
    with pytest.raises(ValueError, match=errmsg):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=T4_protein_component,
            solvent_component=None,
            solvent_offmol=None,
        )


@pytest.mark.parametrize(
    "neutralize, ion_conc",
    [
        [True, 0.0 * unit.molar],
        [False, 0.1 * unit.molar],
        [True, 0.1 * unit.molar],
    ],
)
def test_wrong_solventcomp_settings(neutralize, ion_conc, smc_components_benzene_named):
    with pytest.raises(ValueError, match="Adding counterions"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=neutralize,
                ion_concentration=ion_conc,
            ),
            solvent_offmol=None,
        )


def test_solv_but_no_solv_offmol(
    smc_components_benzene_named,
):
    with pytest.raises(ValueError, match="A solvent offmol"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=None,
        )


def test_solv_mismatch(
    smc_components_benzene_named,
    methanol,
):
    assert ExtendedSolventComponent().smiles == "[H][O][H]"
    with pytest.raises(ValueError, match="does not match"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=methanol,
        )


@pytest.mark.parametrize(
    "assign_charges, errmsg",
    [
        (True, "do not have partial charges"),
        (False, "No library charges"),
    ],
)
def test_charge_assignment_errors(smc_components_benzene_named, assign_charges, errmsg):
    """
    True case: passing a Molecule without partial charges to Interchange
    and asking to get charges from it will fail.
    False case: not having any partial charges will try to see if it's
    in the LibraryCharges if not, it will fail (which it does here).
    """
    solvent_offmol = Molecule.from_smiles("COC")

    with pytest.raises(ValueError, match=errmsg):
        _, _ = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                ]
            ),
            solvation_settings=PackmolSolvationSettings(
                assign_solvent_charges=assign_charges,
            ),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=SolventComponent(
                smiles="COC",
                neutralize=False,
                ion_concentration=0 * unit.molar,
            ),
            solvent_offmol=solvent_offmol,
        )


def test_assign_duplicate_resnames(caplog):
    """
    Pass two smcs named the same and expect one to be renamed
    """
    a = Molecule.from_smiles("C")
    b = Molecule.from_smiles("CCC")
    a.generate_conformers()
    b.generate_conformers()
    a.assign_partial_charges(partial_charge_method="gasteiger")
    b.assign_partial_charges(partial_charge_method="gasteiger")
    _set_offmol_resname(a, "FOO")
    _set_offmol_resname(b, "FOO")
    smc_a = SmallMoleculeComponent.from_openff(a)
    smc_b = SmallMoleculeComponent.from_openff(b)

    smcs = {smc_a: a, smc_b: b}

    with caplog.at_level(logging.WARNING):
        _, smc_comps = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                ]
            ),
            solvation_settings=None,
            smc_components=smcs,
            protein_component=None,
            solvent_component=None,
            solvent_offmol=None,
        )
    for match in ["Duplicate", "residue name to AAA"]:
        assert match in caplog.text

    assert len(smc_comps) == 2
    assert smc_comps[smc_a][0] == 0
    assert smc_comps[smc_b][0] == 1


@pytest.mark.parametrize(
    "smiles",
    [
        "C1CCCCC1",
        "C1CCOC1",
        "CC(=O)N(C)C",
        "CC(=O)c1ccccc1",
        "CC(Br)Br",
        "CC(C)CO",
        "CC(C)O",
        "CC(C)OC(C)C",
        "CC(C)c1ccccc1",
        "CC(Cl)Cl",
        "CCBr",
        "CCCCC",
        "CCCCCC",
        "CCCCCC(C)C",
        "CCCCCCC",
        "CCCCCCCC",
        "CCCCCCCCBr",
        "CCCCCCCl",
        "CCCCCCO",
        "CCCCCO",
        "CCCCO",
        "CCCCOCCCC",
        "CCCCOP(=O)(OCCCC)OCCCC",
        "CCCCc1ccccc1",
        "CCCO",
        "CCN(CC)CC",
        "CCO",
        "CCOCC",
        "CCOc1ccccc1",
        "CCc1ccccc1",
        "CN(C)C=O",
        "CNC=O",
        "COc1ccccc1",
        "Cc1ccccc1",
        "Cc1ccccc1C",
        "Cc1ccccc1C(C)C",
        "Cc1ccccn1",
        "Cc1cccnc1C",
        "c1ccc2c(c1)CCCC2",
        "c1ccccc1",
        "c1ccncc1",
    ],
)
def test_nonwater_solvent(smc_components_benzene_named, smiles):
    solvent_offmol = Molecule.from_smiles(smiles)
    solvent_offmol.assign_partial_charges(partial_charge_method="gasteiger")

    if smiles == "c1ccccc1":
        ligand = list(smc_components_benzene_named.values())[0]
        assert solvent_offmol.is_isomorphic_with(ligand)
        assert_allclose(solvent_offmol.partial_charges, ligand.partial_charges)
        assert all(solvent_offmol.partial_charges == ligand.partial_charges)

    interchange, _ = interchange_packmol_creation(
        ffsettings=InterchangeFFSettings(
            forcefields=[
                "openff-2.0.0.offxml",
            ]
        ),
        solvation_settings=PackmolSolvationSettings(
            assign_solvent_charges=True,
        ),
        smc_components=smc_components_benzene_named,
        protein_component=None,
        solvent_component=SolventComponent(
            smiles=smiles,
            neutralize=False,
            ion_concentration=0 * unit.molar,
        ),
        solvent_offmol=solvent_offmol,
    )

    if smiles == "c1ccccc1":
        # solvent == ligand
        assert solvent_offmol.is_isomorphic_with(
            list(smc_components_benzene_named.values())[0]
        )
        assert interchange.topology.n_unique_molecules == 1
    else:
        assert interchange.topology.n_unique_molecules == 2
        assert solvent_offmol.is_isomorphic_with(
            list(interchange.topology.unique_molecules)[1]
        )


class BaseSystemTests:
    @pytest.fixture(scope="class")
    def omm_system(self, interchange_system):
        interchange, _ = interchange_system

        return interchange.to_openmm_system()

    @pytest.fixture(scope="class")
    def omm_topology(self, interchange_system):
        interchange, _ = interchange_system

        return interchange.to_openmm_topology(collate=True)

    @pytest.fixture(scope="class")
    def nonbonds(self, omm_system):
        return [f for f in omm_system.getForces() if isinstance(f, NonbondedForce)]


class TestVacuumUnamedBenzene(BaseSystemTests):
    smc_comps = "smc_components_benzene_unnamed"
    resname = "AAA"
    nonbond_index = 0

    @pytest.fixture(scope="class")
    def interchange_system(self, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=None,
            solvent_offmol=None,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_particles(self):
        return 12

    @pytest.fixture(scope="class")
    def num_residues(self):
        return 1

    @pytest.fixture(scope="class")
    def num_constraints(self):
        return 6

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 6

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 18

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 42

    def test_comp_resids(self, interchange_system, request):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 1
        assert next(iter(comp_resids.values())) == 0
        assert next(iter(request.getfixturevalue(self.smc_comps))) in comp_resids

    def test_topology(self, omm_topology, num_residues, num_particles):
        residues = list(omm_topology.residues())
        assert len(residues) == num_residues
        assert len(list(omm_topology.atoms())) == num_particles
        assert residues[0].name == self.resname  # Expect auto-naming to AAA
        assert residues[0].index == residues[0].id == 0

    def test_system_basics(self, omm_system, num_particles, num_constraints):
        # Expected number of atoms
        assert omm_system.getNumParticles() == num_particles

        # 4 forces expected
        assert omm_system.getNumForces() == 4

        # 6 constraints, one for each hydrogen
        assert omm_system.getNumConstraints() == num_constraints

    def test_positions(self, interchange_system, num_particles):
        inter, _ = interchange_system
        assert len((to_openmm_positions(inter))) == num_particles

    def test_system_nonbonded(self, nonbonds):
        # One nonbonded force
        assert len(nonbonds) == 1

        # Gas phase should be nonbonded
        assert nonbonds[0].getNonbondedMethod() == self.nonbond_index
        assert nonbonds[0].getCutoffDistance() == to_openmm(0.9 * unit.nanometer)
        assert nonbonds[0].getSwitchingDistance() == to_openmm(0.8 * unit.nanometer)

    def test_system_bonds(self, omm_system, num_bonds):
        bond = [f for f in omm_system.getForces() if isinstance(f, HarmonicBondForce)]

        # One bond forces
        assert len(bond) == 1

        # 6 bonds
        assert bond[0].getNumBonds() == num_bonds

    def test_system_angles(self, omm_system, num_angles):
        angle = [f for f in omm_system.getForces() if isinstance(f, HarmonicAngleForce)]

        # One bond forces
        assert len(angle) == 1

        # 18 angles
        assert angle[0].getNumAngles() == num_angles

    def test_system_dihedrals(self, omm_system, num_dih):
        dih = [f for f in omm_system.getForces() if isinstance(f, PeriodicTorsionForce)]

        # One bond forces
        assert len(dih) == 1

        # 42 angles
        assert dih[0].getNumTorsions() == num_dih


class TestVacuumNamedBenzene(TestVacuumUnamedBenzene):
    smc_comps = "smc_components_benzene_named"
    resname = "BNZ"


class TestSolventOPC3UnamedBenzene(TestVacuumUnamedBenzene):
    smc_comps = "smc_components_benzene_unnamed"
    resname = "AAA"
    nonbond_index = 4
    solvent_resname = "SOL"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_residues(self, omm_topology):
        return omm_topology.getNumResidues()

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues):
        return num_residues - 1

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters):
        return 12 + (3 * num_waters)

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 6 + (3 * num_waters)

    def test_comp_resids(self, interchange_system, request, num_residues):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 2
        assert list(comp_resids)[0] == ExtendedSolventComponent()
        assert list(comp_resids)[1] == next(
            iter(request.getfixturevalue(self.smc_comps))
        )
        assert_equal(list(comp_resids.values())[0], [i for i in range(1, num_residues)])
        assert_equal(list(comp_resids.values())[1], [0])

    def test_solvent_resnames(self, omm_topology):
        for i, res in enumerate(list(omm_topology.residues())[1:]):
            assert res.index == res.id == i + 1
            assert res.name == self.solvent_resname

    def test_solvent_nonbond_parameters(self, nonbonds, num_particles, num_waters):
        for index in range(12, num_particles - num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert from_openmm(c) == -0.89517 * unit.elementary_charge
            assert from_openmm(e).m == pytest.approx(0.683690704)
            assert from_openmm(s).m_as(unit.angstrom) == pytest.approx(
                3.1742703509365926
            )

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert from_openmm(c1) == 0.447585 * unit.elementary_charge
            assert from_openmm(e1) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s1).m == pytest.approx(0.17817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


class TestSolventOPC3NamedChargedButUnAssignedBenzene(TestSolventOPC3UnamedBenzene):
    """
    OPC3 model
    Charged water offmol
    Charges not assigned
    """

    solvent_resname = "HOH"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off_named_charged, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off_named_charged,
        )

        return interchange, comp_resids


class TestSolventOPC3NamedChargedAssignedBenzene(TestSolventOPC3UnamedBenzene):
    """
    OPC3 model
    Charged water offmol
    Charges assigned
    """

    solvent_resname = "HOH"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off_named_charged, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(
                assign_solvent_charges=True,
            ),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off_named_charged,
        )

        return interchange, comp_resids

    def test_solvent_nonbond_parameters(
        self, nonbonds, num_particles, num_waters, water_off_named_charged
    ):
        for index in range(12, num_particles - num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert from_openmm(c) == water_off_named_charged.partial_charges[0]
            assert from_openmm(e).m == pytest.approx(0.683690704)
            assert from_openmm(s).m_as(unit.angstrom) == pytest.approx(
                3.1742703509365926
            )

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert from_openmm(c1) == water_off_named_charged.partial_charges[1]
            assert from_openmm(e1) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s1).m == pytest.approx(0.17817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


class TestSolventOPCNamedBenzene(TestSolventOPC3UnamedBenzene):
    smc_comps = "smc_components_benzene_named"
    resname = "BNZ"
    nonbond_index = 4

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters):
        return 12 + (4 * num_waters)

    def test_solvent_nonbond_parameters(self, nonbonds, num_particles, num_waters):
        for index in range(12, num_particles - num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert from_openmm(c) == 0 * unit.elementary_charge
            assert from_openmm(e).m == pytest.approx(0.890358601)
            assert from_openmm(s).m_as(unit.angstrom) == pytest.approx(3.16655208)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert from_openmm(c1) == 0.679142 * unit.elementary_charge
            assert from_openmm(e1) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s1).m == pytest.approx(0.17817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2

    def test_virtual_sites(self, omm_system, num_waters, num_particles, nonbonds):
        for index in range(num_particles, num_particles - num_waters, -1):
            assert omm_system.isVirtualSite(index - 1)
            c, s, e = nonbonds[0].getParticleParameters(index - 1)
            assert from_openmm(c) == -0.679142 * 2 * unit.elementary_charge
            assert from_openmm(e) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s) * 2 ** (1 / 6) / 2.0 == 1 * unit.angstrom


# def test_setcharge_coc_solvent(smc_components_benzene):
#    ...
#
# def test_inconsistent_solvent_name(smc_components_benzene):
#    ...
#
#
# def test_duplicate_named_smcs(smc_components_benzene):
#    ...
#
#
# def test_box_setting_cube(smc_components_benzene):
#    ...
#
#
# def test_box_setting_dodecahedron(smc_components_benzene):
#    ...


"""
5. Unamed solvent
  - Check we get warned about renaming
6. Named solvent with inconsistent name
7. Duplicate named smcs
10. Cube
11. Dodecahedron
12. Check we get the right residues
13. Check we get the right number of atoms
  - with a solvent w/ virtual sites
  - check omm topology indices match virtual sites (it doesn't!)
14. Check nonbonded cutoffs set via ffsettings
15. Check charged mols tests.
"""
