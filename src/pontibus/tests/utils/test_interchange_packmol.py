# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
import logging
from openmm import (
    NonbondedForce,
    HarmonicBondForce,
    HarmonicAngleForce,
    PeriodicTorsionForce,
)
from openff.toolkit import Molecule, ForceField
from openff.units import unit
from openff.units.openmm import to_openmm, from_openmm
from openff.interchange.interop.openmm import to_openmm_positions
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
    _check_library_charges,
)
from pontibus.utils.molecules import WATER
from numpy.testing import assert_allclose, assert_equal


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


def test_noncharge_nolibrarycharges(smc_components_benzene_named):
    solvent_offmol = Molecule.from_smiles("COC")

    with pytest.raises(ValueError, match="No library charges"):
        _, _ = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[
                    "openff-2.0.0.offxml",
                ]
            ),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=SolventComponent(
                smiles="COC",
                neutralize=False,
                ion_concentration=0 * unit.molar,
            ),
            solvent_offmol=solvent_offmol,
        )


class BaseSystemTests:
    @pytest.fixture(scope="class")
    def omm_system(self, interchange_system):
        interchange, _ = interchange_system

        return interchange.to_openmm_system()

    @pytest.fixture(scope="class")
    def omm_topology(self, interchange_system):
        interchange, _ = interchange_system

        return interchange.to_openmm_topology()

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


class TestSolventOPCNamedBenzene(TestVacuumUnamedBenzene):
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
    def num_residues(self, omm_topology):
        return omm_topology.getNumResidues()

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues):
        return num_residues - 1

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters):
        return 12 + (4 * num_waters)

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 6 + (3 * num_waters)

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 6

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 18

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 42

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
            assert res.name == "SOL"

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

    def test_positions(self, interchange_system, num_particles):
        inter, _ = interchange_system
        assert len((to_openmm_positions(inter))) == num_particles


# def test_library_charges_opc3(smc_components_benzene):
#    ... prenamed, passed on solvent component
#
#
# def test_setcharge_water_solvent(smc_components_benzene):
#    ...
#
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
  - check omm topology indices match virtual sites (it doesn't!)
14. Check nonbonded cutoffs set via ffsettings
"""
