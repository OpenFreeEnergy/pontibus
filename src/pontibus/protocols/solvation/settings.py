# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using OpenMM.

See Also
--------
openfe.protocols.openmm_afe.AbsoluteSolvationProtocol
"""
from typing import (
    Optional,
    Literal,
)
from openff.units import unit
from openff.models.types import FloatQuantity
from gufe.settings import (
    BaseForceFieldSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    MultiStateSimulationSettings,
    BaseSolvationSettings,
    OpenMMEngineSettings,
    IntegratorSettings,
    OpenFFPartialChargeSettings,
    MultiStateOutputSettings,
    MDSimulationSettings,
    MDOutputSettings,
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AlchemicalSettings,
    LambdaSettings,
    AbsoluteSolvationSettings,
)
from pydantic.v1 import validator


class InterchangeFFSettings(BaseForceFieldSettings):
    """
    Parameters to set up the force field using Interchange and the
    OpenFF toolkit.
    """

    hydrogen_mass: float = 3.0
    """Mass to be repartitioned to hydrogens from neighbouring
       heavy atoms (in amu), default 3.0"""

    # TODO; work out if we could swap these out with str of ffxml contents
    # if necessary
    forcefields: list[str] = [
        "openff-2.0.0.offxml",
        "tip3p.offxml",
    ]
    """List of force field ffxmls to apply"""

    nonbonded_method = "PME", "NoCutoff"
    """
    Method for treating nonbonded interactions, currently only PME and
    NoCutoff are allowed. Default PME.
    """

    nonbonded_cutoff: FloatQuantity["nanometer"] = 1.0 * unit.nanometer
    """
    Cutoff value for short range nonbonded interactions.
    Default 1.0 * unit.nanometer.
    """

    @validator("nonbonded_method")
    def allowed_nonbonded(cls, v):
        # TODO: switch to literal?
        if v.lower() not in ["pme", "nocutoff"]:
            errmsg = "Only PME and NoCutoff are allowed nonbonded_methods"
            raise ValueError(errmsg)
        return v

    @validator("nonbonded_cutoff")
    def is_positive_distance(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.nanometer):
            raise ValueError(
                "nonbonded_cutoff must be in distance units " "(i.e. nanometers)"
            )
        if v < 0:
            errmsg = "nonbonded_cutoff must be a positive value"
            raise ValueError(errmsg)
        return v


class PackmolSolvationSettings(BaseSolvationSettings):
    """
    Settings defining how to solvate the system using Packmol.

    Notes
    -----
    * This is currently limited to the options allowed by
      Interchange's ``solvate_topology_nonwater``.
    """

    solvent_padding: Optional[FloatQuantity["nanometer"]] = 1.2 * unit.nanometer
    """
    Minimum distance from any solute bounding sphere to the edge of the box.

    """

    box_shape: Optional[Literal["cube", "dodecahedron"]] = "cube"
    """
    The shape of the periodic box to create.
    """

    assign_solvent_charges: bool = False
    """
    If ``True``, assign solvent charges based on the input solvent
    molecule. If ``False``, rely on library charges.

    Notes
    -----
    * If no partial charges are set in the input molecule, the molecule
    will be charged using the approach defined in ``partial_charge_settings``.
    * If not using ``ExtendedSolventComponent``, the input molecule will
    be created using ``SolventComponent.smiles`` and partial charges will
    be set using the approach defined in ``partial_charge_settings``.
    """


class ASFESettings(AbsoluteSolvationSettings):
    """
    Configuration object for ``ASFEProtocol``.

    See Also
    --------
    pontibus.protocols.solvation.ASFEProtocol
    """

    # Inherited things
    solvent_forcefield_settings: InterchangeFFSettings
    """Parameters to set up in the solvent force field"""

    vacuum_forcefield_settings: InterchangeFFSettings
    """Parameters to set up the vacuum force field"""

    solvation_settings: PackmolSolvationSettings
    """Settings for solvating the system."""
