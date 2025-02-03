# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using OpenMM.

See Also
--------
openfe.protocols.openmm_afe.AbsoluteSolvationProtocol
"""
from typing import Literal, Optional

from gufe.settings import BaseForceFieldSettings, ThermoSettings
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteSolvationSettings,
    AlchemicalSettings,
    LambdaSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BaseSolvationSettings,
    IntegratorSettings,
    MDOutputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
)
from openff.models.types import FloatQuantity
from openff.units import unit
from pydantic.v1 import validator


class ExperimentalAlchemicalSettings(AlchemicalSettings):
    experimental: bool = False
    """
    Turn on experimental alchemy settings
    """


class InterchangeFFSettings(BaseForceFieldSettings):
    """
    Parameters to set up the force field using Interchange and the
    OpenFF toolkit.
    """

    hydrogen_mass: float = 3.023841
    """Mass to be repartitioned to hydrogens from neighbouring
       heavy atoms (in amu), default 3.0"""

    # TODO; work out if we could swap these out with str of ffxml contents
    # if necessary
    forcefields: list[str] = [
        "openff-2.0.0.offxml",
        "tip3p.offxml",
    ]
    """List of force field ffxmls to apply"""

    nonbonded_method: Literal["pme", "nocutoff"] = "pme"
    """
    Method for treating nonbonded interactions, currently only PME and
    NoCutoff are allowed. Default PME.
    """

    nonbonded_cutoff: FloatQuantity["nanometer"] = 0.9 * unit.nanometer
    """
    Cutoff value for short range nonbonded interactions.
    Default 1.0 * unit.nanometer.
    """

    switch_width: FloatQuantity["nanometer"] = 0.1 * unit.nanometer
    """
    The width over which the VdW switching function is applied.
    Default 0.1 * unit.nanometer.
    """

    @validator("nonbonded_method")
    def allowed_nonbonded(cls, v):
        # TODO: switch to literal?
        if v.lower() not in ["pme", "nocutoff"]:
            errmsg = "Only PME and NoCutoff are allowed nonbonded_methods"
            raise ValueError(errmsg)
        return v

    @validator("nonbonded_cutoff", "switch_width")
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

    packing_tolerance: FloatQuantity["angstrom"] = 0.75 * unit.angstrom
    """
    Packmol setting; minimum spacing between molecules in units of distance.
    2.0 A is recommended when packing proteins, but can go as low as 0.5 A
    to help with convergence.
    """

    target_density: FloatQuantity["grams / mL"] = 0.95 * unit.grams / unit.mL
    """
    Target mass density for the solvated system in units compatible with g / mL.
    Generally a ``target_density`` value of 0.95 * unit.grams / unit.mL is
    sufficient, although you may have to aim for a lower value should you find
    it difficult to pack your system.

    Default: 0.95 * unit.grams / unit.mL.
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
