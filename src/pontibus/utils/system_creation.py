# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
from itertools import product
from string import ascii_uppercase
from typing import Any

import numpy as np
import numpy.typing as npt
from gufe import Component, ProteinComponent, SmallMoleculeComponent, SolventComponent
from openff.interchange import Interchange
from openff.interchange.components._packmol import (
    RHOMBIC_DODECAHEDRON,
    UNIT_CUBE,
    pack_box,
    solvate_topology_nonwater,
)
from openff.toolkit import ForceField, Topology
from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit as offunit

from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)

logger = logging.getLogger(__name__)


def _set_offmol_resname(
    offmol: OFFMolecule,
    resname: str,
) -> None:
    """
    Helper method to set offmol residue names

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to assign a residue name to.
    resname : str
      Residue name to be set.

    Returns
    -------
    None
    """
    for a in offmol.atoms:
        a.metadata["residue_name"] = resname


def _get_offmol_resname(offmol: OFFMolecule) -> str | None:
    """
    Helper method to get an offmol's residue name and make sure it is
    consistent across all atoms in the Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to get the residue name from.

    Returns
    -------
    resname : Optional[str]
      Residue name of the molecule. ``None`` if the Molecule
      does not have a residue name, or if the residue name is
      inconsistent across all the atoms.
    """
    resname: str | None = None
    for a in offmol.atoms:
        if resname is None:
            try:
                resname = a.metadata["residue_name"]
            except KeyError:
                return None

        if resname != a.metadata["residue_name"]:
            wmsg = f"Inconsistent residue name in OFFMol: {offmol} "
            logger.warning(wmsg)
            return None

    return resname


def _check_library_charges(
    force_field: ForceField,
    offmol: OFFMolecule,
) -> None:
    """
    Check that library charges exists for an input molecule.

    force_field : openff.toolkit.ForceField
      Force Field object with library charges.
    offmol : openff.toolkit.Molecule
      Molecule to check for matching library charges.

    Raises
    ------
    ValueError
      If no library charges are found for the molecule.
    """
    handler = force_field.get_parameter_handler("LibraryCharges")
    matches = handler.find_matches(offmol.to_topology())

    if len(matches) == 0:
        errmsg = f"No library charges found for {offmol}"
        raise ValueError(errmsg)


def _check_and_deduplicate_charged_mols(
    molecules: list[OFFMolecule],
) -> list[OFFMolecule]:
    """
    Checks list of molecules with charges and removes any isomorphic
    duplicates so that it can be passed to Interchange for partial
    charge assignment.

    Parameters
    ----------
    molecules : list[openff.toolkit.Molecule]
      A list of molecules with charges.

    Returns
    -------
    unique_mols : list[openff.toolkit.Molecule]
      A list of ismorphically unique molecules with charges.

    Raises
    ------
    ValueError
      If any molecules in the list are isomorphic with different charges.
      If any molecules in the last have no charges.
    """
    if any(m.partial_charges is None for m in molecules):
        errmsg = (
            "One or more molecules have been explicitly passed "
            "for partial charge assignment but do not have "
            "partial charges"
        )
        raise ValueError(errmsg)

    unique_mols = []

    for moli in molecules:
        isomorphic_mols = [molj for molj in unique_mols if moli.is_isomorphic_with(molj)]

        if isomorphic_mols:
            # If we have any cases where there are isomorphic mols
            # either:
            # 1. They have the same charge so we don't add a second entry
            # 2. They have different charges and it's an error.
            for molj in isomorphic_mols:
                if not all(moli.partial_charges == molj.partial_charges):
                    errmsg = (
                        f"Isomorphic molecules {moli} and {molj}"
                        "have been passed for partial charge "
                        "assignment with different charges. "
                        "This is not currently allowed."
                    )
                    raise ValueError(errmsg)
        else:
            unique_mols.append(moli)

    return unique_mols


def _validate_components(
    protein_component: ProteinComponent | None,
    solvent_component: SolventComponent | None,
    solvent_offmol: OFFMolecule | None,
) -> None:
    """
    Validates input components to ``interchange_packmol_creation``.


    Parameters
    ----------
    protein_component : Optional[ProteinComponent]
      Protein component of the system, if any.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary


    Raises
    ------
    ValueError
      If there is a protein_component.
      If the solvent_component requests counterions.
      If we have a solvent_component but no solvent_offmol.
      If the solvent_component and solvent_offmol are not isomorphic.
      If the solvent_offmol doesn't have one conformer.
    """
    # Adding protein components is not currently supported
    if protein_component is not None:
        errmsg = (
            "Creation of systems solely with Interchange "
            "using ProteinComponents is not currently supported"
        )
        raise ValueError(errmsg)

    # TODO: work out ways to deal with the addition of counterions
    if solvent_component is not None:
        if solvent_component.neutralize or solvent_component.ion_concentration > 0 * offunit.molar:
            errmsg = "Adding counterions using packmol solvation is currently not supported"
            raise ValueError(errmsg)

        if solvent_offmol is None:
            errmsg = "A solvent offmol must be passed to solvate a system!"
            raise ValueError(errmsg)

        # Check that the component matches the offmol
        if not solvent_offmol.is_isomorphic_with(OFFMolecule.from_smiles(solvent_component.smiles)):
            errmsg = (
                f"Passed molecule: {solvent_offmol} does not match the "
                f"the solvent component: {solvent_component.smiles}"
            )
            raise ValueError(errmsg)

        # TODO: check here that the offmol has a single conformer
        if solvent_offmol.n_conformers != 1:
            errmsg = (
                "Solvent OpenFF Molecule should have a single conformer. "
                f"Number of conformers found: {solvent_offmol.n_conformers}"
            )
            raise ValueError(errmsg)


def _get_force_field(ffsettings: InterchangeFFSettings) -> ForceField:
    """
    Get a ForceField object based on an input InterchangeFFSettings object.

    Parameters
    ----------
    ffsettings : InterchangeFFSettings
      Settings defining how the force field is applied.

    Returns
    -------
    force_field : ForceField
      An OpenFF toolkit ForceField object.
    """
    # forcefields is a list so we unpack it
    force_field = ForceField(*ffsettings.forcefields)

    # Cautiously deregister the AM1BCC handler, we shouldn't need it.
    # See: https://github.com/openforcefield/openff-interchange/issues/1048
    force_field.deregister_parameter_handler("ToolkitAM1BCC")

    # We also set nonbonded cutoffs whilst we are here
    # TODO: double check what this means for nocutoff simulations
    force_field["Electrostatics"].cutoff = ffsettings.nonbonded_cutoff
    force_field["vdW"].cutoff = ffsettings.nonbonded_cutoff
    force_field["vdW"].switch_width = ffsettings.switch_width

    return force_field


def _get_comp_resnames(
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    solvent_component: SolventComponent | None,
    solvent_offmol: OFFMolecule | None,
) -> dict[str, tuple[Component, list[Any]]]:
    """
    Assign residue names so we can track components in a generated Topology.

    Parameters
    ----------
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]`
      Solute SmallMoleculeComponents.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary

    Returns
    -------
    comp_resnames: dict[str, tuple[Component, list[Any]]]
      A dictionary keyed by residue names which contains
      a tuple with the matching Component and an empty list
      which will later be populated with residue numbers.
    """
    # Note: comp_resnames is dict[str, tuple[Component, list]] where the final
    # list is to append residues later on
    # TODO: make this a method
    # TODO: we should be able to rely on offmol equality in the same way that
    # intechange does
    comp_resnames: dict[str, tuple[Component, list[Any]]] = {}

    # If we have solvent, we set its residue name
    if solvent_component is not None:
        offmol_resname = _get_offmol_resname(solvent_offmol)
        if offmol_resname is None:
            offmol_resname = "SOL"
            _set_offmol_resname(solvent_offmol, offmol_resname)
        comp_resnames[offmol_resname] = (solvent_component, [])

    # A store of residue names to replace residue names if they aren't unique
    resnames_store = ["".join(i) for i in product(ascii_uppercase, repeat=3)]

    for comp, offmol in smc_components.items():
        off_resname = _get_offmol_resname(offmol)
        if off_resname is None or off_resname in comp_resnames:
            # warn that we are overriding clashing molecule resnames
            if off_resname in comp_resnames:
                wmsg = f"Duplicate residue name {off_resname}, duplicate will be renamed"
                logger.warning(wmsg)

            # just loop through and pick up a name that doesn't exist
            while (off_resname in comp_resnames) or (off_resname is None):
                off_resname = resnames_store.pop(0)

        wmsg = f"Setting component {comp} residue name to {off_resname}"
        logger.warning(wmsg)
        _set_offmol_resname(offmol, off_resname)
        comp_resnames[off_resname] = [comp, []]

    return comp_resnames


def _solvate_system(
    solute_topology: Topology,
    solvent_offmol: OFFMolecule,
    solvation_settings: PackmolSolvationSettings,
) -> Topology:
    """
    Solvate solute Topology using the Interchange packmol interface.

    Parameters
    ----------
    solute_topology : Topology
      The solute Topology to solvate.
    solvent_offmol : OFFMolecule
      An OpenFF Molecule representing the solvent.
    solvation_settings : PackmolSolvationSettings
      Settings for how to solvate the system.

    Returns
    -------
    Topology
      The solvated Topology.
    """
    # Pick up the user selected box shape
    box_shape = {
        "cube": UNIT_CUBE,
        "dodecahedron": RHOMBIC_DODECAHEDRON,
    }[solvation_settings.box_shape.lower()]

    # Create the topology
    if solvation_settings.number_of_solvent_molecules is not None:
        return pack_box(
            molecules=[solvent_offmol],
            number_of_copies=[solvation_settings.number_of_solvent_molecules],
            solute=solute_topology,
            tolerance=solvation_settings.packing_tolerance,
            box_vectors=solvation_settings.box_vectors,
            target_density=solvation_settings.target_density,
            box_shape=box_shape,
            center_solute=True,
            working_directory=None,
            retain_working_files=False,
        )
    else:
        return solvate_topology_nonwater(
            topology=solute_topology,
            solvent=solvent_offmol,
            target_density=solvation_settings.target_density,
            padding=solvation_settings.solvent_padding,
            box_shape=box_shape,
            tolerance=solvation_settings.packing_tolerance,
        )


def interchange_packmol_creation(
    ffsettings: InterchangeFFSettings,
    solvation_settings: PackmolSolvationSettings,
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    protein_component: ProteinComponent | None,
    solvent_component: SolventComponent | None,
    solvent_offmol: OFFMolecule | None,
) -> tuple[Interchange, dict[str, npt.NDArray]]:
    """
    Create an Interchange object for a given system, solvating with
    packmol where necessary.

    Parameters
    ----------
    ffsettings : InterchangeFFSettings
      Settings defining how the force field is applied.
    solvation_settings : PackmolSolvationSettings
      Settings defining how the system will be solvated.
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]`
      Solute SmallMoleculeComponents.
    protein_component : Optional[ProteinComponent]
      Protein component of the system, if any.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary

    Returns
    -------
    Interchange : openff.interchange.Interchange
      Interchange object for the created system.
    comp_resids : dict[str, npt.NDArray]
      A dictionary defining the residue indices matching
      various components in the system.
    """

    # 1. Component validations
    _validate_components(protein_component, solvent_component, solvent_offmol)

    # 2. Get the force field object
    force_field = _get_force_field(ffsettings)

    # 3. Assign residue names so we can track our components in the generated
    # topology.
    comp_resnames = _get_comp_resnames(smc_components, solvent_component, solvent_offmol)

    # 4. Create an OFF Topology from the smcs
    # Note: this is the base no solvent case!
    topology = Topology.from_molecules([*smc_components.values()])

    # Also create a list of charged molecules for later use
    charged_mols = [*smc_components.values()]

    # 5. Solvent case
    if solvent_component is not None:
        # Append to charged molcule to charged_mols if we want to
        # otherwise we rely on library charges
        if solvation_settings.assign_solvent_charges:
            charged_mols.append(solvent_offmol)
        else:
            # Make sure we have library charges for the molecule
            _check_library_charges(force_field, solvent_offmol)

        topology = _solvate_system(
            topology,
            solvent_offmol,
            solvation_settings,
        )

    # TODO: maybe make this a method
    # Assign residue indices to each entry in the OFF topology
    for molecule_index, molecule in enumerate(topology.molecules):
        for atom in molecule.atoms:
            atom.metadata["residue_number"] = molecule_index

        # Get the residue name and store the index in comp resnames
        resname = _get_offmol_resname(molecule)
        comp_resnames[resname][1].append(molecule_index)

    # Now create the component_resids dictionary properly
    comp_resids = {}
    for entry in comp_resnames.values():
        comp = entry[0]
        comp_resids[comp] = np.array(entry[1])

    # Run validation checks on inputs to Interchange
    # Examples: https://github.com/openforcefield/openff-interchange/issues/1058
    unique_charged_mols = _check_and_deduplicate_charged_mols(charged_mols)

    interchange = force_field.create_interchange(
        topology=topology,
        charge_from_molecules=unique_charged_mols,
    )

    return interchange, comp_resids
