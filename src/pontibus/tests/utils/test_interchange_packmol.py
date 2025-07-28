# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import logging

import numpy as np
import pytest
from gufe import SmallMoleculeComponent, SolventComponent
from numpy.testing import assert_allclose, assert_equal
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import ForceField, Molecule, Topology
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import (
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)
from rdkit import Chem

from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.molecule_utils import (
    _check_library_charges,
    _get_offmol_resname,
    _set_offmol_resname,
)
from pontibus.utils.molecules import WATER
from pontibus.utils.system_creation import (
    _check_and_deduplicate_charged_mols,
    _get_comp_resnames,
    _get_force_field,
    _solvate_system,
    interchange_packmol_creation,
    _proteincomp_to_topology,
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


@pytest.fixture(scope="module")
def smc_components_acetic_acid():
    mol = Molecule.from_smiles("CC(=O)[O-]")
    mol.generate_conformers(n_conformers=1)
    mol.assign_partial_charges(partial_charge_method="gasteiger")
    return {SmallMoleculeComponent.from_openff(mol): mol}


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


def test_convert_proteincomp(T4_protein_component):
    # Get an OpenFF Toplogy by going through rdkit
    rdmols = Chem.GetMolFrags(T4_protein_component.to_rdkit(), asMols=True, sanitizeFrags=False)
    ofe_mol = Molecule.from_rdkit(rdmols[0], allow_undefined_stereo=True, hydrogens_are_explicit=True)
    ofe_top = Topology.from_molecules([ofe_mol])
    # The the OpenFF Topology with the tooling
    off_top = _proteincomp_to_topology(T4_protein_component)

    # light isormophic check
    assert off_top.molecule(0).is_isomorphic_with(ofe_top.molecule(0), atom_stereochemistry_matching=False)


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
    assert "Inconsistent metadata residue_name" in caplog.text


def test_check_library_charges_pass(water_off):
    ff = ForceField("opc.offxml")
    _check_library_charges(ff, water_off)


def test_check_library_charges_fail(methanol):
    ff = ForceField("openff-2.0.0.offxml")
    with pytest.raises(ValueError, match="No library charges"):
        _check_library_charges(ff, methanol)


def test_check_charged_mols_pass(methanol):
    _check_and_deduplicate_charged_mols([methanol])


def test_check_deduplicated_charged_mols(smc_components_benzene_unnamed):
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
def test_wrong_solventcomp_settings_nonwater(
    neutralize, ion_conc, smc_components_benzene_named, methanol
):
    with pytest.raises(ValueError, match="Counterions are currently not"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=neutralize,
                ion_concentration=ion_conc,
                solvent_molecule=SmallMoleculeComponent.from_openff(methanol),
            ),
            solvent_offmol=methanol,
        )


def test_not_neutralize_but_ion_conc(
    smc_components_benzene_named,
    water_off,
):
    with pytest.raises(ValueError, match="Cannot add ions without"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=False,
                ion_concentration=0.1 * unit.molar,
            ),
            solvent_offmol=water_off,
        )


@pytest.mark.parametrize("pos, neg", [["Na+", "F-"], ["K+", "Cl-"], ["K+", "F-"]])
def test_bad_ions(
    smc_components_benzene_named,
    water_off,
    pos,
    neg,
):
    with pytest.raises(ValueError, match="Can only neutralize with NaCl"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=True,
                ion_concentration=0 * unit.molar,
                positive_ion=pos,
                negative_ion=neg,
            ),
            solvent_offmol=water_off,
        )


@pytest.mark.parametrize("resname", ["NA+", "CL-"])
def test_resname_solvent_ion_clash(smc_components_benzene_named, resname):
    solv_off = WATER.to_openff()
    _set_offmol_resname(solv_off, resname)

    errmsg = "Solvent resname is set to"
    with pytest.raises(ValueError, match=errmsg):
        _get_comp_resnames(
            smc_components_benzene_named, ExtendedSolventComponent(neutralize=True), solv_off
        )


def test_solvate_system_neutralize_nonwater(methanol):
    msg = "Cannot neutralize a system with non-water solvent"
    with pytest.raises(ValueError, match=msg):
        _solvate_system(
            solute_topology=Topology(),
            solvent_offmol=methanol,
            solvation_settings=PackmolSolvationSettings(),
            neutralize=True,
            ion_concentration=0.1 * unit.molar,
        )


def test_solvate_system_neutralize_num_sol_defined(water_off):
    msg = "Cannot neutralize a system where the number of waters"
    with pytest.raises(ValueError, match=msg):
        _solvate_system(
            solute_topology=Topology(),
            solvent_offmol=water_off,
            solvation_settings=PackmolSolvationSettings(
                number_of_solvent_molecules=100,
                solvent_padding=None,
            ),
            neutralize=True,
            ion_concentration=0.1 * unit.molar,
        )


def test_solvate_system_neutralize_bad_conc(water_off):
    msg = "is not compatible with mole / liter"
    with pytest.raises(ValueError, match=msg):
        _solvate_system(
            solute_topology=Topology(),
            solvent_offmol=water_off,
            solvation_settings=PackmolSolvationSettings(),
            neutralize=True,
            ion_concentration=0.1 * unit.nm,
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


def test_no_solvent_conformers(
    smc_components_benzene_named,
):
    solmol = Molecule.from_smiles("C")
    solmol.assign_partial_charges(partial_charge_method="gasteiger")
    solmol.generate_conformers()
    solvent = ExtendedSolventComponent(solvent_molecule=SmallMoleculeComponent.from_openff(solmol))
    solmol._conformers = []

    with pytest.raises(ValueError, match="single conformer"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(assign_solvent_charges=True),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=solvent,
            solvent_offmol=solmol,
        )


def test_get_force_field():
    ffsettings = InterchangeFFSettings(
        nonbonded_cutoff=1.0 * unit.nanometer,
        switch_width=0.2 * unit.nanometer,
    )

    ff = _get_force_field(ffsettings)

    assert ff["vdW"].cutoff == ff["Electrostatics"].cutoff == 1.0 * unit.nanometer
    assert ff["vdW"].switch_width == 0.2 * unit.nanometer


def test_get_force_field_custom():
    """
    Re-implementation of the test Josh implemented in OMMForcefields
    """
    sage = ForceField("openff-2.2.1.offxml")
    ethane = Molecule.from_smiles("C")
    bond_parameter = sage.label_molecules(ethane.to_topology())[0]["Bonds"][(0, 1)]
    bonds = sage.get_parameter_handler("Bonds")
    new_parameter = bonds[bond_parameter.smirks]
    new_parameter.length = 2 * unit.angstrom

    ffsettings = InterchangeFFSettings(forcefields=[sage.to_string(), "opc.offxml"])

    ff = _get_force_field(ffsettings)

    bonds = ff.get_parameter_handler("Bonds")
    bond_param = bonds[bond_parameter.smirks]
    assert bond_param.length == 2 * unit.angstrom


def test_multiple_solvent_conformers(
    smc_components_benzene_named,
):
    solmol = Molecule.from_smiles("CCCCCCCCCCCCCCCCCCCCCCCCCCC")
    solmol.generate_conformers()
    solmol.assign_partial_charges(partial_charge_method="gasteiger")
    solvent = ExtendedSolventComponent(solvent_molecule=SmallMoleculeComponent.from_openff(solmol))

    with pytest.raises(ValueError, match="single conformer"):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(assign_solvent_charges=True),
            smc_components=smc_components_benzene_named,
            protein_component=None,
            solvent_component=solvent,
            solvent_offmol=solmol,
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
    solvent_offmol.generate_conformers(n_conformers=1)

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
def test_nonwater_solvent_short(smc_components_benzene_named, smiles):
    solvent_offmol = Molecule.from_smiles(smiles)
    solvent_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    solvent_offmol.generate_conformers(n_conformers=1)

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
            solvent_padding=None,
            number_of_solvent_molecules=100,
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
        assert solvent_offmol.is_isomorphic_with(list(smc_components_benzene_named.values())[0])
        assert interchange.topology.n_unique_molecules == 1
    else:
        assert interchange.topology.n_unique_molecules == 2
        assert solvent_offmol.is_isomorphic_with(list(interchange.topology.unique_molecules)[1])
    assert interchange.topology.n_molecules == 101


@pytest.mark.cpuvslow
@pytest.mark.parametrize(
    "solvent_smiles, solute_smiles",
    [
        ("Cc1ccccn1", "CCCCCCCC"),
        ("Cc1ccccn1", "Cc1ccccc1"),
        ("Cc1ccccn1", "CCO"),
        ("Cc1ccccn1", "C1COCCO1"),
        ("Cc1ccccn1", "CCC(C)=O"),
        ("CC(=O)c1ccccc1", "CCCCCCCC"),
        ("CC(=O)c1ccccc1", "Cc1ccccc1"),
        ("CC(=O)c1ccccc1", "CCO"),
        ("CC(=O)c1ccccc1", "C1COCCO1"),
        ("CC(=O)c1ccccc1", "CCC(C)=O"),
        ("COc1ccccc1", "CCCCCCCC"),
        ("COc1ccccc1", "Cc1ccccc1"),
        ("COc1ccccc1", "CCO"),
        ("COc1ccccc1", "C1COCCO1"),
        ("COc1ccccc1", "CCC(C)=O"),
        ("COc1ccccc1", "CCCCN"),
        ("c1ccccc1", "CCCCCCCC"),
        ("c1ccccc1", "C1CCCCC1"),
        ("c1ccccc1", "CC(C)O"),
        ("c1ccccc1", "CC(C)(C)O"),
        ("c1ccccc1", "C1COCCO1"),
        ("c1ccccc1", "CC(C)=O"),
        ("c1ccccc1", "CCC(C)=O"),
        ("c1ccccc1", "COC(C)=O"),
        ("c1ccccc1", "CNC"),
        ("c1ccccc1", "CN(C)C"),
        ("c1ccccc1", "c1ccncc1"),
        ("c1ccccc1", "Cc1cccc(C)n1"),
        ("c1ccccc1", "O=Cc1cccc(O)c1"),
        ("c1ccccc1", "O=Cc1ccc(O)cc1"),
        ("c1ccccc1", "Oc1ccc(Br)cc1"),
        ("c1ccccc1", "CCOP(=O)(OCC)OCC"),
        ("c1ccccc1", "C1CCNCC1"),
        ("c1ccccc1", "CN"),
        ("c1ccccc1", "COC(=O)c1ccccc1"),
        ("c1ccccc1", "COP(=O)(OC)OC=C(Cl)Cl"),
        ("c1ccccc1", "Cc1cncc(C)c1"),
        ("c1ccccc1", "NC(=O)c1ccccc1"),
        ("c1ccccc1", "Cc1ccccc1N"),
        ("CCBr", "CCCCCCCC"),
        ("CCBr", "Cc1ccccc1"),
        ("CCBr", "C1COCCO1"),
        ("CCBr", "CCC(C)=O"),
        ("CCCCCCCCBr", "COC(C)=O"),
        ("CCCCCCCCBr", "CCCCCOC(C)=O"),
        ("CCCCO", "CCCCCCCC"),
        ("CCCCO", "c1ccccc1"),
        ("CCCCO", "CO"),
        ("CCCCO", "CCO"),
        ("CCCCO", "C=O"),
        ("CCCCO", "CCC(C)=O"),
        ("CCCCO", "CCN"),
        ("CCCCO", "CCCN"),
        ("CCCCO", "CCNCC"),
        ("CCCCO", "O=c1cc[nH]c(=O)[nH]1"),
        ("CCCCc1ccccc1", "Oc1ccccc1"),
        ("CCCCc1ccccc1", "CC(=O)C(C)(C)C"),
        ("CCCCc1ccccc1", "CCCCCC(C)=O"),
        ("CCCCc1ccccc1", "CCC(=O)OC"),
        ("CCCCc1ccccc1", "CCCCOC(C)=O"),
        ("CCCCCCCl", "CC(C)=O"),
        ("CCCCCCCl", "CC(=O)C(C)(C)C"),
        ("CCCCCCCl", "CCC(=O)OC"),
        ("CCCCCCCl", "CCCCOC(C)=O"),
        ("C1CCCCC1", "CCC"),
        ("C1CCCCC1", "CCCCCCCC"),
        ("C1CCCCC1", "CC(C)O"),
        ("C1CCCCC1", "CC(C)(C)O"),
        ("C1CCCCC1", "C1COCCO1"),
        ("C1CCCCC1", "COc1ccccc1"),
        ("C1CCCCC1", "CCC(=O)CC"),
        ("C1CCCCC1", "CC(=O)C(C)(C)C"),
        ("C1CCCCC1", "COC(C)=O"),
        ("C1CCCCC1", "CCN"),
        ("C1CCCCC1", "CN(C)C"),
        ("C1CCCCC1", "CCNCC"),
        ("C1CCCCC1", "c1ccncc1"),
        ("C1CCCCC1", "CNc1ccccc1"),
        ("C1CCCCC1", "CCCS"),
        ("C1CCCCC1", "CSc1ccccc1"),
        ("C1CCCCC1", "O=Cc1cccc(O)c1"),
        ("C1CCCCC1", "O=Cc1ccc(O)cc1"),
        ("C1CCCCC1", "Clc1ccccc1"),
        ("C1CCCCC1", "Clc1ccc(Cl)cc1"),
        ("C1CCCCC1", "Brc1ccccc1"),
        ("C1CCCCC1", "Oc1ccc(Br)cc1"),
        ("C1CCCCC1", "COP(=O)(OC)OC"),
        ("C1CCCCC1", "CCOP(=O)(OCC)OCC"),
        ("C1CCCCC1", "COC(=O)c1ccccc1"),
        ("C1CCCCC1", "C1CCOCC1"),
        ("C1CCCCC1", "N#Cc1cc(Br)c(O)c(Br)c1"),
        ("C1CCCCC1", "CN(C)C=O"),
        ("C1CCCCC1", "c1cc[nH]c1"),
        ("C1CCCCC1", "NC(=O)c1ccccc1"),
        ("C1CCCCC1", "Cc1ccccc1N"),
        ("CC(Br)Br", "CO"),
        ("CC(Br)Br", "CCCCO"),
        ("CC(Br)Br", "Oc1ccccc1"),
        ("CC(Br)Br", "Oc1ccc(Br)cc1"),
        ("CCCCOCCCC", "CCCCCCCC"),
        ("CCCCOCCCC", "CCO"),
        ("CCCCOCCCC", "C1COCCO1"),
        ("CCCCOCCCC", "CCC(C)=O"),
        ("CCCCOCCCC", "CCNCC"),
        ("CCCCOCCCC", "c1ccncc1"),
        ("CCCCOCCCC", "Cc1cnccn1"),
        ("CCCCOCCCC", "Cc1ccccn1"),
        ("CCCCOCCCC", "CCc1cnccn1"),
        ("CC(Cl)Cl", "CO"),
        ("CC(Cl)Cl", "CCCCO"),
        ("CC(Cl)Cl", "Cc1ccccc1O"),
        ("CC(Cl)Cl", "O=Cc1ccccc1"),
        ("CC(Cl)Cl", "CC(=O)c1ccccc1"),
        ("CC(Cl)Cl", "COC(C)=O"),
        ("CC(Cl)Cl", "CCCCCC(=O)OC"),
        ("CC(Cl)Cl", "CCN"),
        ("CC(Cl)Cl", "CCCN"),
        ("CC(Cl)Cl", "CCNCC"),
        ("CC(Cl)Cl", "c1ccncc1"),
        ("CC(Cl)Cl", "O=Cc1cccc(O)c1"),
        ("CC(Cl)Cl", "Oc1ccc(Br)cc1"),
        ("CC(Cl)Cl", "COP(=O)(OC)OC"),
        ("CC(Cl)Cl", "CCOP(=O)(OCC)OCC"),
        ("CC(Cl)Cl", "NC(=O)c1ccccc1"),
        ("CCOCC", "CCCCCCCC"),
        ("CCOCC", "CC(C)O"),
        ("CCOCC", "CC(C)(C)O"),
        ("CCOCC", "C1COCCO1"),
        ("CCOCC", "COc1ccccc1"),
        ("CCOCC", "CC=O"),
        ("CCOCC", "CCC(C)=O"),
        ("CCOCC", "CC(=O)c1ccccc1"),
        ("CCOCC", "CNC"),
        ("CCOCC", "CN(C)C"),
        ("CCOCC", "c1ccncc1"),
        ("CCOCC", "Nc1ccccc1"),
        ("CCOCC", "C=CCO"),
        ("CCOCC", "COCCO"),
        ("CCOCC", "O=Cc1cccc(O)c1"),
        ("CCOCC", "Clc1ccccc1"),
        ("CCOCC", "Clc1ccc(Cl)cc1"),
        ("CCOCC", "Brc1ccccc1"),
        ("CCOCC", "S"),
        ("CCOCC", "C1CCNCC1"),
        ("CCOCC", "CN"),
        ("CCOCC", "CC(N)=O"),
        ("CCOCC", "CN(C)C=O"),
        ("CCOCC", "NC=O"),
        ("CCOCC", "O=c1cc[nH]c(=O)[nH]1"),
        ("CCOCC", "O=c1[nH]cc(Br)c(=O)[nH]1"),
        ("CC(C)OC(C)C", "CCCCCCCC"),
        ("CC(C)OC(C)C", "c1ccc2ccccc2c1"),
        ("CC(C)OC(C)C", "CCO"),
        ("CC(C)OC(C)C", "C1COCCO1"),
        ("CC(C)OC(C)C", "C=O"),
        ("CC(C)OC(C)C", "CCC(C)=O"),
        ("CC(C)OC(C)C", "CN(C)C"),
        ("CC(C)OC(C)C", "CCNCC"),
        ("CC(C)OC(C)C", "c1ccncc1"),
        ("CC(C)OC(C)C", "Nc1ccccc1"),
        ("CC(C)OC(C)C", "O=Cc1ccc(O)cc1"),
        ("CC(=O)N(C)C", "CCCCCCCC"),
        ("CC(=O)N(C)C", "Cc1ccccc1"),
        ("CC(=O)N(C)C", "CCO"),
        ("CC(=O)N(C)C", "C1COCCO1"),
        ("CC(=O)N(C)C", "CCC(C)=O"),
        ("CN(C)C=O", "CCCCCCCC"),
        ("CN(C)C=O", "Cc1ccccc1"),
        ("CN(C)C=O", "CCO"),
        ("CN(C)C=O", "C1COCCO1"),
        ("CN(C)C=O", "CCC(C)=O"),
        ("Cc1cccnc1C", "CCCCCCCC"),
        ("Cc1cccnc1C", "Cc1ccccc1"),
        ("Cc1cccnc1C", "CCO"),
        ("Cc1cccnc1C", "C1COCCO1"),
        ("Cc1cccnc1C", "CCC(C)=O"),
        ("Cc1cccnc1C", "Cc1cccc(C)n1"),
        ("CCO", "CCCCCCCC"),
        ("CCO", "Cc1ccccc1"),
        ("CCO", "C1COCCO1"),
        ("CCO", "CCC(C)=O"),
        ("CCO", "Clc1ccccc1"),
        ("CCO", "O=C1CCCO1"),
        ("CCOc1ccccc1", "CCCCCCCC"),
        ("CCOc1ccccc1", "Cc1ccccc1"),
        ("CCOc1ccccc1", "CCO"),
        ("CCOc1ccccc1", "C1COCCO1"),
        ("CCOc1ccccc1", "CCC(C)=O"),
        ("CCc1ccccc1", "CO"),
        ("CCc1ccccc1", "CCCCCO"),
        ("CCc1ccccc1", "Cc1ccccc1O"),
        ("CCc1ccccc1", "CCC(C)=O"),
        ("CCc1ccccc1", "CC(=O)C(C)(C)C"),
        ("CCc1ccccc1", "COC(C)=O"),
        ("CCc1ccccc1", "CCCCC(=O)OC"),
        ("CCc1ccccc1", "CCN"),
        ("CCc1ccccc1", "CCCN"),
        ("CCc1ccccc1", "CN(C)C"),
        ("CCc1ccccc1", "Oc1ccc(Br)cc1"),
        ("CCCCCCC", "CO"),
        ("CCCCCCC", "CCCCCO"),
        ("CCCCCCC", "COc1ccccc1"),
        ("CCCCCCC", "O=Cc1ccccc1"),
        ("CCCCCCC", "CCC(C)=O"),
        ("CCCCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCCCC", "COC(C)=O"),
        ("CCCCCCC", "CCCCC(=O)OC"),
        ("CCCCCCC", "CCN"),
        ("CCCCCCC", "c1ccncc1"),
        ("CCCCCCC", "Clc1ccc(Cl)cc1"),
        ("CCCCCCC", "Brc1ccccc1"),
        ("CCCCCCC", "Brc1ccc(Br)cc1"),
        ("CCCCCCC", "COP(=O)(OC)OC"),
        ("CCCCCCC", "CCOP(=O)(OCC)OCC"),
        ("CCCCCCC", "c1ccsc1"),
        ("CCCCCCC", "Clc1ccccc1-c1ccccc1Cl"),
        ("CCCCCCC", "CC(=O)N(C)C"),
        ("CCCCCCC", "NC(=O)c1ccccc1"),
        ("CCCCCCC", "Cc1ccccc1N"),
        ("CCCCCCC", "CN1CCCC1=O"),
        ("CCCCCCC", "O=c1[nH]cc(Br)c(=O)[nH]1"),
        ("CCCCCC", "CCCCCCCC"),
        ("CCCCCC", "CO"),
        ("CCCCCC", "CCO"),
        ("CCCCCC", "C1COCCO1"),
        ("CCCCCC", "O=Cc1ccccc1"),
        ("CCCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCCC", "CC(=O)c1ccccc1"),
        ("CCCCCC", "COC(C)=O"),
        ("CCCCCC", "CCCCC(=O)OC"),
        ("CCCCCC", "CCN"),
        ("CCCCCC", "c1ccncc1"),
        ("CCCCCC", "O=Cc1ccc(O)cc1"),
        ("CCCCCC", "Clc1ccccc1"),
        ("CCCCCC", "Clc1ccc(Cl)cc1"),
        ("CCCCCC", "Brc1ccccc1"),
        ("CCCCCC", "Oc1ccc(Br)cc1"),
        ("CCCCCC", "COP(=O)(OC)OC"),
        ("CCCCCC", "CCOP(=O)(OCC)OCC"),
        ("CCCCCC", "N#Cc1cc(Br)c(O)c(Br)c1"),
        ("CCCCCC", "NC(=O)c1ccccc1"),
        ("CCCCCC", "Cc1ccc(N)cc1"),
        ("CCCCCCO", "c1ccccc1"),
        ("CCCCCCO", "Oc1ccccc1"),
        ("CCCCCCO", "C=O"),
        ("CCCCCCO", "CCN"),
        ("CCCCCCO", "CCCCN"),
        ("CCCCCCO", "Oc1ccc(Br)cc1"),
        ("CC(C)CO", "CCCC=O"),
        ("CC(C)CO", "CCOC(C)=O"),
        ("CC(C)CO", "CNC"),
        ("CC(C)CO", "CN(C)C"),
        ("CC(C)CO", "C1CNCCN1"),
        ("CC(C)CO", "c1ccncc1"),
        ("CC(C)CO", "C1CCNCC1"),
        ("CC(C)CO", "CN"),
        ("CCCCCC(C)C", "CCCCC"),
        ("CCCCCC(C)C", "CCCCCC"),
        ("CCCCCC(C)C", "C=CC"),
        ("CCCCCC(C)C", "C=CCC"),
        ("CCCCCC(C)C", "c1ccccc1"),
        ("CCCCCC(C)C", "CCO"),
        ("CCCCCC(C)C", "CCCCCCO"),
        ("CCCCCC(C)C", "Cc1ccccc1O"),
        ("CCCCCC(C)C", "C1COCCO1"),
        ("CCCCCC(C)C", "CCCC=O"),
        ("CCCCCC(C)C", "CCCCC=O"),
        ("CCCCCC(C)C", "CC(C)=O"),
        ("CCCCCC(C)C", "CCC(C)=O"),
        ("CCCCCC(C)C", "CCCCN"),
        ("CCCCCC(C)C", "Nc1ccccc1"),
        ("CCCCCC(C)C", "CCS"),
        ("CCCCCC(C)C", "CCCS"),
        ("CCCCCC(C)C", "COC(=O)c1ccccc1"),
        ("CC(C)O", "CCCCCCCC"),
        ("CC(C)O", "Cc1ccccc1"),
        ("CC(C)O", "CCO"),
        ("CC(C)O", "C1COCCO1"),
        ("CC(C)O", "CCC(C)=O"),
        ("CC(C)c1ccccc1", "CCO"),
        ("CC(C)c1ccccc1", "CC(C)=O"),
        ("CC(C)c1ccccc1", "CC(=O)C(C)(C)C"),
        ("CC(C)c1ccccc1", "CCOC(C)=O"),
        ("CC(C)c1ccccc1", "CCCCCC(=O)OC"),
        ("CC(C)c1ccccc1", "CCCCN"),
        ("Cc1ccccc1C(C)C", "CCC(=O)OC"),
        ("Cc1ccccc1C(C)C", "CCCCCOC(C)=O"),
        ("Cc1ccccc1C(C)C", "CCCCN"),
        ("CNC=O", "CCCCCCCC"),
        ("CNC=O", "Cc1ccccc1"),
        ("CNC=O", "CCO"),
        ("CNC=O", "C1COCCO1"),
        ("CNC=O", "CCC(C)=O"),
        ("CCCCCCCC", "CO"),
        ("CCCCCCCC", "CCCO"),
        ("CCCCCCCC", "CCC(C)=O"),
        ("CCCCCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCCCCC", "COC(C)=O"),
        ("CCCCCCCC", "CCCCCC(=O)OC"),
        ("CCCCCCCC", "CCN"),
        ("CCCCCCCC", "CCNCC"),
        ("CCCCCCCC", "Cc1cnccn1"),
        ("CCCCCCCC", "Nc1ccccc1"),
        ("CCCCCCCC", "CCc1cnccn1"),
        ("CCCCCCCC", "Cc1ccccc1N"),
        ("CCCCC", "CO"),
        ("CCCCC", "CCCO"),
        ("CCCCC", "Oc1ccccc1"),
        ("CCCCC", "CCCC(C)=O"),
        ("CCCCC", "CC(=O)C(C)(C)C"),
        ("CCCCC", "COC(C)=O"),
        ("CCCCC", "CCCCCC(=O)OC"),
        ("CCCCC", "CCN"),
        ("CCCCC", "CCCN"),
        ("CCCCC", "Nc1ccccc1"),
        ("CCCCCO", "c1ccccc1"),
        ("CCCCCO", "Cc1ccccc1O"),
        ("CCCCCO", "C=O"),
        ("CCCCCO", "CCCCN"),
        ("CCCCCO", "CCNCC"),
        ("CCCCCO", "Oc1ccc(Br)cc1"),
        ("CCCCCO", "CN"),
        ("CCCO", "CCCCCCCC"),
        ("CCCO", "Cc1ccccc1"),
        ("CCCO", "CCO"),
        ("CCCO", "C1COCCO1"),
        ("CCCO", "CCC(C)=O"),
        ("c1ccncc1", "CCCCCCCC"),
        ("c1ccncc1", "Cc1ccccc1"),
        ("c1ccncc1", "CCO"),
        ("c1ccncc1", "C1COCCO1"),
        ("c1ccncc1", "CCC(C)=O"),
        ("C1CCOC1", "CCCCCCCC"),
        ("C1CCOC1", "Cc1ccccc1"),
        ("C1CCOC1", "CCO"),
        ("C1CCOC1", "C1COCCO1"),
        ("C1CCOC1", "CCC(C)=O"),
        ("c1ccc2c(c1)CCCC2", "CCO"),
        ("c1ccc2c(c1)CCCC2", "CC(=O)C(C)(C)C"),
        ("c1ccc2c(c1)CCCC2", "CCCCCC(C)=O"),
        ("Cc1ccccc1", "CCCCCCCC"),
        ("Cc1ccccc1", "CO"),
        ("Cc1ccccc1", "CCO"),
        ("Cc1ccccc1", "C1COCCO1"),
        ("Cc1ccccc1", "CC(C)=O"),
        ("Cc1ccccc1", "CC(=O)C(C)(C)C"),
        ("Cc1ccccc1", "COC(C)=O"),
        ("Cc1ccccc1", "CNC"),
        ("Cc1ccccc1", "CN(C)C"),
        ("Cc1ccccc1", "CCCNCCC"),
        ("Cc1ccccc1", "c1ccncc1"),
        ("Cc1ccccc1", "Nc1ccccc1"),
        ("Cc1ccccc1", "Oc1ccc(Br)cc1"),
        ("Cc1ccccc1", "CN"),
        ("Cc1ccccc1", "O=C1CCCO1"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CO"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CCCCCO"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CCN"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "CCCN"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "Nc1ccccc1"),
        ("CCCCOP(=O)(OCCCC)OCCCC", "COCCO"),
        ("CCN(CC)CC", "CCCCCCCC"),
        ("CCN(CC)CC", "Cc1ccccc1"),
        ("CCN(CC)CC", "CCO"),
        ("CCN(CC)CC", "C1COCCO1"),
        ("CCN(CC)CC", "CCC(C)=O"),
        ("Cc1ccccc1C", "CCCCCCCC"),
        ("Cc1ccccc1C", "CO"),
        ("Cc1ccccc1C", "CCO"),
        ("Cc1ccccc1C", "Cc1ccc(O)cc1"),
        ("Cc1ccccc1C", "C1COCCO1"),
        ("Cc1ccccc1C", "CC(C)=O"),
        ("Cc1ccccc1C", "CC(=O)C(C)(C)C"),
        ("Cc1ccccc1C", "COC(C)=O"),
        ("Cc1ccccc1C", "CCCCC(=O)OC"),
        ("Cc1ccccc1C", "CNC"),
        ("Cc1ccccc1C", "CN(C)C"),
        ("Cc1ccccc1C", "c1ccncc1"),
        ("Cc1ccccc1C", "Nc1ccccc1"),
        ("Cc1ccccc1C", "Oc1ccc(Br)cc1"),
        ("Cc1ccccc1C", "C1CCNCC1"),
        ("Cc1ccccc1C", "CN"),
    ],
)
def test_nonwater_solvent_long(solvent_smiles, solute_smiles):
    solvent_offmol = Molecule.from_smiles(solvent_smiles)
    solvent_offmol.generate_conformers(n_conformers=1)
    solvent_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    solvent_smc = SmallMoleculeComponent.from_openff(solvent_offmol)

    ligand_offmol = Molecule.from_smiles(solute_smiles)
    ligand_offmol.generate_conformers(n_conformers=1)
    ligand_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    ligand_smc = SmallMoleculeComponent.from_openff(ligand_offmol)

    interchange, _ = interchange_packmol_creation(
        ffsettings=InterchangeFFSettings(
            forcefields=[
                "openff-2.0.0.offxml",
            ]
        ),
        solvation_settings=PackmolSolvationSettings(
            solvent_padding=None,
            number_of_solvent_molecules=1000,
            assign_solvent_charges=True,
        ),
        smc_components={ligand_smc: ligand_offmol},
        protein_component=None,
        solvent_component=ExtendedSolventComponent(solvent_molecule=solvent_smc),
        solvent_offmol=solvent_offmol,
    )

    if solvent_smiles == solute_smiles:
        # solvent == ligand
        assert solvent_offmol.is_isomorphic_with(ligand_offmol)
        assert interchange.topology.n_unique_molecules == 1
    else:
        assert interchange.topology.n_unique_molecules == 2
        assert solvent_offmol.is_isomorphic_with(list(interchange.topology.unique_molecules)[1])
    assert interchange.topology.n_molecules == 1001


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

        # 5 forces expected
        # Note: CMMotionRemover is there and needs to be removed in Protocol
        assert omm_system.getNumForces() == 5

        # 6 constraints, one for each hydrogen
        assert omm_system.getNumConstraints() == num_constraints

    def test_positions(self, interchange_system, num_particles):
        inter, _ = interchange_system
        assert len(to_openmm_positions(inter)) == num_particles

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
        assert list(comp_resids)[1] == next(iter(request.getfixturevalue(self.smc_comps)))
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
            assert from_openmm(s).m_as(unit.angstrom) == pytest.approx(3.1742703509365926)

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
            assert from_openmm(s).m_as(unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert from_openmm(c1) == water_off_named_charged.partial_charges[1]
            assert from_openmm(e1) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s1).m == pytest.approx(0.17817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2


class TestSolventOPC3AceticAcidNeutralize(TestSolventOPC3UnamedBenzene):
    smc_comps = "smc_components_acetic_acid"

    @pytest.fixture(scope="class")
    def interchange_system(self, water_off, request):
        smc_components = request.getfixturevalue(self.smc_comps)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=["openff-2.0.0.offxml", "opc3.offxml"],
            ),
            solvation_settings=PackmolSolvationSettings(
                solvent_padding=2 * unit.nm,
            ),
            smc_components=smc_components,
            protein_component=None,
            solvent_component=ExtendedSolventComponent(
                neutralize=True,
                ion_concentration=0.15 * unit.molar,
            ),
            solvent_offmol=water_off,
        )

        return interchange, comp_resids

    @pytest.fixture(scope="class")
    def num_bonds(self):
        return 3

    @pytest.fixture(scope="class")
    def num_angles(self):
        return 9

    @pytest.fixture(scope="class")
    def num_dih(self):
        return 15

    @pytest.fixture(scope="class")
    def num_pos_ions(self):
        return 7

    @pytest.fixture(scope="class")
    def num_neg_ions(self):
        return 6

    @pytest.fixture(scope="class")
    def num_waters(self, num_residues, num_pos_ions, num_neg_ions):
        return num_residues - (1 + num_neg_ions + num_pos_ions)

    @pytest.fixture(scope="class")
    def num_particles(self, num_waters, num_neg_ions, num_pos_ions):
        return 7 + (3 * num_waters) + num_neg_ions + num_pos_ions

    @pytest.fixture(scope="class")
    def num_constraints(self, num_waters):
        return 3 + (3 * num_waters)

    def test_comp_resids(self, interchange_system, request, num_residues):
        _, comp_resids = interchange_system

        assert len(comp_resids) == 2
        assert list(comp_resids)[0] == ExtendedSolventComponent(
            neutralize=True,
            ion_concentration=0.15 * unit.molar,
        )
        assert list(comp_resids)[1] == next(iter(request.getfixturevalue(self.smc_comps)))
        assert_equal(list(comp_resids.values())[0], [i for i in range(1, num_residues)])
        assert_equal(list(comp_resids.values())[1], [0])

    def test_solvent_resnames(self, omm_topology):
        for i, res in enumerate(list(omm_topology.residues())[1:]):
            assert res.index == res.id == i + 1
            assert res.name in [self.solvent_resname, "NA+", "CL-"]

    def test_solvent_nonbond_parameters(self, nonbonds, num_particles, num_waters):
        for index in range(7, 7 + num_waters, 3):
            # oxygen
            c, s, e = nonbonds[0].getParticleParameters(index)
            assert from_openmm(c) == -0.89517 * unit.elementary_charge
            assert from_openmm(e).m == pytest.approx(0.683690704)
            assert from_openmm(s).m_as(unit.angstrom) == pytest.approx(3.1742703509365926)

            # hydrogens
            c1, s1, e1 = nonbonds[0].getParticleParameters(index + 1)
            c2, s2, e2 = nonbonds[0].getParticleParameters(index + 2)
            assert from_openmm(c1) == 0.447585 * unit.elementary_charge
            assert from_openmm(e1) == 0 * unit.kilocalorie_per_mole
            assert from_openmm(s1).m == pytest.approx(0.17817974)
            assert c1 == c2
            assert s1 == s2
            assert e2 == e2

        for index in range(7 + (num_waters * 3), num_particles):
            c, s, e = nonbonds[0].getParticleParameters(index)

            charge = from_openmm(c)
            assert abs(charge.m) == 1

            if charge.m == 1:
                assert from_openmm(e).m == pytest.approx(0.1260287744)
                assert from_openmm(s).m_as(unit.angstroms) == pytest.approx(2.617460434)

            if charge.m == -1:
                assert from_openmm(e).m == pytest.approx(2.68724395648)
                assert from_openmm(s).m_as(unit.angstroms) == pytest.approx(4.108824888)

    def test_system_total_charge(self, nonbonds, omm_system):
        total_charge = 0.0
        for i in range(omm_system.getNumParticles()):
            c, s, e = nonbonds[0].getParticleParameters(i)
            total_charge += from_openmm(c).m

        assert total_charge == pytest.approx(0)


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
