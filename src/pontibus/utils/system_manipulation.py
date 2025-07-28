# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import numpy as np
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology
from openmm import Force, System

from pontibus.utils.molecule_utils import (
    _get_offmol_metadata,
    _set_offmol_metadata,
)
from pontibus.utils.system_creation import _check_and_deduplicate_charged_mols


def adjust_system(
    system: System,
    remove_force_types: type | list[type] | None = None,
    add_forces: Force | list[Force] | None = None,
) -> None:
    """
    Adjust a System by removing and adding forces as necessary.

    Parameters
    ----------
    system : System
      The OpenMM System to adjust
    remove_force_types : type | list[type] | None
      The force types to remove from the System, if present.
    add_forces : list[Force] | None
      The forces to add to the system.
    """

    def _adjust_inputs(var):
        if var is not None:
            if isinstance(var, list):
                return var
            return [var]
        else:
            return []

    remove_force_types = _adjust_inputs(remove_force_types)
    add_forces = _adjust_inputs(add_forces)

    for entry in remove_force_types:  # type: ignore[union-attr]
        for idx in reversed(range(system.getNumForces())):
            force = system.getForce(idx)
            if isinstance(force, entry):
                system.removeForce(idx)

    for force in add_forces:
        system.addForce(force)


def copy_interchange_with_replacement(
    interchange: Interchange,
    del_mol: Molecule,
    insert_mol: Molecule,
    force_field: ForceField,
    charged_molecules: list[Molecule] | None,
) -> Interchange:
    """
    Copy an Interchange deleting one Molecule and appending another.

    Parameters
    ----------
    interchange : Interchange
      Input Interchange to copy.
    del_mol : Molecule
      The Molecule to delete from the Interchange.
    insert_mol : Molecule
      The Molecule to insert to the Interchange.
    force_field : ForceField
      The ForceField object used to create the initial Interchange.
    charged_molecules : list[Molecule] | None
      A  list of Molecules which partial charges to use in the new Interchange.

    Returns
    -------
    new_interchange : Interchange
      An copy of the input Interchange but with the Molecule mutation.

    Note
    ----
    * ``del_mol`` is always deleted and ``insert_mol`` is appended to the end.
    * The residue number of the Molecule matching ``del_mol`` in the input
      Interchange is transcribed over to the ``insert_mol`` molecule.
    """
    # Validate
    if insert_mol.conformers is None or del_mol.conformers is None:
        raise ValueError("Input molecules need conformers")

    # Get the del_mol idx
    del_mol_idx = None

    # Search the Interchange Topology for a molecule with both
    # isomorphic and spatial equality.
    for idx, mol in enumerate(interchange.topology.molecules):
        if mol.is_isomorphic_with(del_mol):
            if np.allclose(mol.conformers[0], del_mol.conformers[0]):
                if del_mol_idx is not None:
                    raise ValueError("equality clash with del_mol")

                del_mol_idx = idx
                del_mol_resnum = _get_offmol_metadata(mol, "residue_number")

    if del_mol_idx is None:
        errmsg = "No Molecule matching del_mol in input Interchange"
        raise ValueError(errmsg)

    # Set molB residue number to molA
    _set_offmol_metadata(insert_mol, "residue_number", del_mol_resnum)

    # Get a list of molecules from the input Interchange
    mols = [m for m in interchange.topology.molecules]
    mols.pop(del_mol_idx)  # pop out the Molecule to be deleted
    mols.append(insert_mol)  # insert the new Molecule

    new_topology = Topology.from_molecules(mols)
    new_topology.box_vectors = interchange.topology.box_vectors

    if charged_molecules is not None:
        charged_molecules = _check_and_deduplicate_charged_mols(charged_molecules)

    new_interchange = force_field.create_interchange(
        topology=new_topology,
        charge_from_molecules=charged_molecules,
    )

    return new_interchange
