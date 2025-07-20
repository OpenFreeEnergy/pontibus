# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using OpenMM.

See Also
--------
openfe.protocols.openmm_afe.AbsoluteSolvationProtocol
"""

from typing import Literal

from gufe.settings import BaseForceFieldSettings
from gufe.vendor.openff.models.types import ArrayQuantity, FloatQuantity
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteSolvationSettings,
    AlchemicalSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BaseSolvationSettings,
)
from openff.interchange.components._packmol import _box_vectors_are_in_reduced_form
from openff.units import unit
from pydantic.v1 import root_validator, validator


class ExperimentalAlchemicalSettings(AlchemicalSettings):
    experimental: bool = False
    """
    Enable the use of experimental alchemical features.

    This includes, but is not limited to, support for virtual sites
    in alchemical transformations.
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

    nonbonded_cutoff: FloatQuantity["nanometer"] = 0.9 * unit.nanometer  # noqa: F821
    """
    Cutoff value for short range nonbonded interactions.
    Default 1.0 * unit.nanometer.
    """

    switch_width: FloatQuantity["nanometer"] = 0.1 * unit.nanometer  # noqa: F821
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
            raise ValueError("nonbonded_cutoff must be in distance units (i.e. nanometers)")
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

    number_of_solvent_molecules: int | None = None
    """
    The number of solvent molecules to add.

    Note
    ----
    * Cannot be defined alongside ``solvent_padding``.
    """

    box_vectors: ArrayQuantity["nanometer"] | None = None  # noqa: F821
    """
    Simulation box vectors.

    Note
    ----
    * Cannot be defined alongside ``target_density``.
    * If defined, ``number_of_solvent_molecules`` must be defined.
    """

    solvent_padding: FloatQuantity["nanometer"] | None = 1.2 * unit.nanometer  # noqa: F821
    """
    Minimum distance from any solute bounding sphere to the edge of the box.

    Note
    ----
    * Cannot be defined if ``number_of_solvent_molecules`` is defined.
    """

    box_shape: Literal["cube", "dodecahedron"] | None = "cube"
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

    packing_tolerance: FloatQuantity["angstrom"] = 2.0 * unit.angstrom  # noqa: F821
    """
    Packmol setting; minimum spacing between molecules in units of distance.
    2.0 A is recommended when packing proteins, but can go as low as 0.5 A
    to help with convergence.
    """

    target_density: FloatQuantity["grams / mL"] | None = 0.95 * unit.grams / unit.mL  # noqa: F821
    """
    Target mass density for the solvated system in units compatible with g / mL.
    Generally a ``target_density`` value of 0.95 * unit.grams / unit.mL is
    sufficient, although you may have to aim for a lower value should you find
    it difficult to pack your system.

    Default: 0.95 * unit.grams / unit.mL.

    Notes
    -----
    * Cannot be defined alongside ``box_vectors``
    """

    @validator("number_of_solvent_molecules")
    def positive_solvent_number(cls, v):
        if v is None:
            return v

        if v <= 0:
            errmsg = f"number_of_solvent molecules: {v} must be positive"
            raise ValueError(errmsg)

        return v

    @validator("box_vectors")
    def supported_vectors(cls, v):
        if v is not None:
            if not _box_vectors_are_in_reduced_form(v):
                errmsg = f"box_vectors: {v} are not in OpenMM reduced form"
                raise ValueError(errmsg)
        return v

    @root_validator
    def check_num_mols_or_padding(cls, values):
        num_solvent = values.get("number_of_solvent_molecules")
        padding = values.get("solvent_padding")

        if not (num_solvent is None) ^ (padding is None):
            msg = "Only one of ``number_solvent_molecules`` or ``solvent_padding`` can be defined"
            raise ValueError(msg)

        return values

    @root_validator
    def check_target_density_or_box_vectors(cls, values):
        target_density = values.get("target_density")
        box_vectors = values.get("box_vectors")

        if not (target_density is None) ^ (box_vectors is None):
            msg = "Only one of ``target_density`` or ``box_vectors`` can be defined"
            raise ValueError(msg)

        return values

    @root_validator
    def check_target_density_and_box_shape(cls, values):
        target_density = values.get("target_density")
        box_shape = values.get("box_shape")

        if not (target_density is None) == (box_shape is None):
            msg = "``target_density`` and ``box_shape`` must both be defined"
            raise ValueError(msg)

        return values

    @root_validator
    def check_solvent_padding_or_box_vectors(cls, values):
        box_vectors = values.get("box_vectors")
        padding = values.get("solvent_padding")

        if box_vectors is not None and (padding is None):
            msg = "Only one of ``box_vectors`` or ``solvent_padding`` can be defined."
            raise ValueError(msg)

        return values


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
