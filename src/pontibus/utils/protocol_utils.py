# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""
Reusable methods for pontibus Protocols.
"""

from openff.toolkit import Molecule as OFFMolecule
from openfe import SolventComponent
from openfe.protocols.openmm_utils import charge_generation
from openfe.protocols.openmm_utils.omm_settings import OpenFFPartialChargeSettings
from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.utils.settings import PackmolSolvationSettings


def _get_and_charge_solvent_offmol(
    solvent_component: SolventComponent | ExtendedSolventComponent,
    solvation_settings: PackmolSolvationSettings,
    partial_charge_settings: OpenFFPartialChargeSettings,
) -> OFFMolecule:
    """
    Helper method to fetch the solvent offmol either
    from an existing solvent_smcs, or from smiles.

    Parameters
    ----------
    solvent_component : SolventComponent
      smiles for the solvent molecule
    solvation_settings : PackmolSolvationSettings
      Settings defining how the system will be solvated
    partial_charge_settings : OpenFFPartialChargeSettigns
      Settings defining how partial charges are applied

    Returns
    -------
    offmol : openff.toolkit.Molecule

    Notes
    -----
    * If created from a smiles, the solvent will be assigned
      a single conformer through `Molecule.generate_conformers`.
    """
    # Get the solvent offmol
    if isinstance(solvent_component, ExtendedSolventComponent):
        solvent_offmol = solvent_component.solvent_molecule.to_openff()  # type: ignore[union-attr]
    else:
        # If not, we create the solvent from smiles
        # We generate a single conformer to avoid packing issues
        solvent_offmol = OFFMolecule.from_smiles(solvent_component.smiles)
        solvent_offmol.generate_conformers(n_conformers=1)

    # In-place assign solvent offmol charges if necessary
    # Note: we don't enforce partial charge assignment to avoid
    # cases where we want to rely on library charges instead.
    if solvation_settings.assign_solvent_charges:
        charge_generation.assign_offmol_partial_charges(
            offmol=solvent_offmol,
            overwrite=False,
            method=partial_charge_settings.partial_charge_method,
            toolkit_backend=partial_charge_settings.off_toolkit_backend,
            generate_n_conformers=partial_charge_settings.number_of_conformers,
            nagl_model=partial_charge_settings.nagl_model,
        )

    return solvent_offmol
