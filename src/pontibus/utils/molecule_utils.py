# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import logging
from typing import Any

from openff.toolkit import ForceField
from openff.toolkit import Molecule as OFFMolecule

logger = logging.getLogger(__name__)


def _set_offmol_metadata(
    offmol: OFFMolecule,
    key: Any,
    val: Any | None,
) -> None:
    """
    Set a given metadata entry for a whole Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      The Molecule to set the metadata for.
    key : Any
      The metadata key.
    val : Any
      The value to set the metadata entry to.
    """
    if val is None:
        for a in offmol.atoms:
            a.metadata.pop(key, None)
    else:
        for a in offmol.atoms:
            a.metadata[key] = val


def _get_offmol_metadata(offmol: OFFMolecule, key: Any) -> Any | None:
    """
    Get an offmol's given metadata entry and make sure it is
    consistent across all atoms in the Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to get the metadata value from.
    key: Any
      The metadata entry key.

    Returns
    -------
    value : Any | None
      Metadata for the given key in the molecule. ``None`` if the
      Molecule does not have that metadata entry set, or if
      the value is inconsistent across all the atoms.
    """
    value: Any | None = None
    for a in offmol.atoms:
        if value is None:
            try:
                value = a.metadata[key]
            except KeyError:
                return None

        if value != a.metadata[key]:
            wmsg = f"Inconsistent metadata {key} in OFFMol: {offmol}"
            logger.warning(wmsg)
            return None

    return value


def _set_offmol_resname(
    offmol: OFFMolecule,
    resname: str | None,
) -> None:
    """
    Helper method to set offmol residue names

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to assign a residue name to.
    resname : str | None
      Residue name to be set. Set to None to clear it.

    Returns
    -------
    None
    """
    _set_offmol_metadata(offmol, "residue_name", resname)


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
    return _get_offmol_metadata(offmol, "residue_name")


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


def _get_num_residues(offmol: OFFMolecule) -> int:
    """
    Get the number of residues in a Molecule based
    on how Interchange creates OpenMM Topology residues.
    """

    num_residx = 0
    last_chain_id = None
    last_resnum = None
    last_resname = None

    for at in offmol.atoms:
        at_resname = at.metadata.get("residue_name", "UNK")
        at_resnum = at.metadata.get("residue_number", "0")
        at_chain_id = at.metadata.get("chain_id", "X")

        if not all(
            (
                (last_resname == at_resname),
                (last_resnum == at_resnum),
                (last_chain_id == at_chain_id),
            )
        ):
            last_chain_id = at_chain_id
            last_resnum = at_resnum
            last_resname = at_resname

            num_residx += 1

    return num_residx


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

    unique_mols: list[OFFMolecule] = []

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
