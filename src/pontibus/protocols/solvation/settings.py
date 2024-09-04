# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using OpenMM.

See Also
--------
openfe.protocols.openmm_afe.AbsoluteSolvationProtocol

TODO
----
* Add support for restraints

"""
from gufe.settings import (
    SettingsBaseModel,
    BaseForceFieldSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    MultiStateSimulationSettings,
    BaseSolvationSettings,
    OpenMMSolvationSettings,
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

)
import numpy as np

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

    nonbonded_method = 'PME', 'NoCutoff'
    """
    Method for treating nonbonded interactions, currently only PME and
    NoCutoff are allowed. Default PME.
    """

    nonbonded_cutoff: FloatQuantity['nanometer'] = 1.0 * unit.nanometer
    """
    Cutoff value for short range nonbonded interactions.
    Default 1.0 * unit.nanometer.
    """

    @validator('nonbonded_method')
    def allowed_nonbonded(cls, v):
        # TODO: switch to literal?
        if v.lower() not in ['pme', 'nocutoff']:
            errmsg = (
                "Only PME and NoCutoff are allowed nonbonded_methods")
            raise ValueError(errmsg)
        return v

    @validator('nonbonded_cutoff')
    def is_positive_distance(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.nanometer):
            raise ValueError("nonbonded_cutoff must be in distance units "
                             "(i.e. nanometers)")
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
    solvent_padding: Optional[FloatQuantity['nanometer']] = 1.2 * unit.nanometer
    """
    Minimum distance from any solute bounding sphere to the edge of the box.

    """

    box_shape: Optional[Literal['cube', 'dodecahedron']] = 'cube'
    """
    The shape of the periodic box to create.
    """

    solvation_mode: Literatl['water', 'nonwater']
    """
    How solvation will happen.

    
    ``water``: Interchange's ``solvate_topology`` will be used. The main
    use case is adding water and ions around a solute. You should use this
    option if you are relying on library charges, e.g. if you are directly
    specifying a water offxml.

    ``nonwater``: Interchange's ``solvate_topology_nonwater`` will be used.
    The main use case is adding non-water molecules around a solute (although
    we note that a water molecule could be passed too!). Here, a
    ``SmallMoleculeComponent`` representing the solvent is expected to be
    passed through in the ``ChemicalSystem`` with the index key set by
    ``small_molecule_key``.

    Note
    ----
    ``small_molecule_key`` will be ignored if the solvation mode is set to ``water``.
    """

    small_molecule_key: Optional[str] = None
    """
    A reserved key in the ChemicalSystem to assign a SmallMoleculeComponent
    as the solvent.

    Note
    ----
    * This should only be set if ``solvation_mode`` is set to ``nonwater``.
      If it is set to ``water``, the SmallMoleculeComponent passed through
      will be ignored.
    """


# This subclasses from SettingsBaseModel as it has vacuum_forcefield and
# solvent_forcefield fields, not just a single forcefield_settings field
class ASFESettings(SettingsBaseModel):
    """
    Configuration object for ``AbsoluteSolvationProtocol``.

    See Also
    --------
    openfe.protocols.openmm_afe.AbsoluteSolvationProtocol
    """
    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling 
    process. The mean of the repeats defines the final estimate of FE 
    difference, while the variance between repeats is used as the uncertainty.  
    """

    @validator('protocol_repeats')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # Inherited things
    solvent_forcefield_settings: InterchangeFFSettings
    vacuum_forcefield_settings: InterchangeFFSettings
    """Parameters to set up the force field with OpenMM Force Fields"""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters"""

    solvation_settings: PackmolSolvationSettings
    """Settings for solvating the system."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings.
    """
    lambda_settings: LambdaSettings
    """
    Settings for controlling the lambda schedule for the different components 
    (vdw, elec, restraints).
    """

    # MD Engine things
    vacuum_engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform
    for the vacuum transformation.
    """
    solvent_engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform
    for the solvent transformation.
    """

    # Sampling State defining things
    integrator_settings: IntegratorSettings
    """
    Settings for controlling the integrator, such as the timestep and
    barostat settings.
    """

    # Simulation run settings
    vacuum_equil_simulation_settings: MDSimulationSettings
    """
    Pre-alchemical vacuum simulation control settings.

    Notes
    -----
    The `NVT` equilibration should be set to 0 * unit.nanosecond
    as it will not be run.
    """
    vacuum_simulation_settings: MultiStateSimulationSettings
    """
    Simulation control settings, including simulation lengths
    for the vacuum transformation.
    """
    solvent_equil_simulation_settings: MDSimulationSettings
    """
    Pre-alchemical solvent simulation control settings.
    """
    solvent_simulation_settings: MultiStateSimulationSettings
    """
    Simulation control settings, including simulation lengths
    for the solvent transformation.
    """
    vacuum_equil_output_settings: MDOutputSettings
    """
    Simulation output settings for the vacuum non-alchemical equilibration.
    """
    vacuum_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the vacuum transformation.
    """
    solvent_equil_output_settings: MDOutputSettings
    """
    Simulation output settings for the solvent non-alchemical equilibration.
    """
    solvent_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the solvent transformation.
    """
    partial_charge_settings: OpenFFPartialChargeSettings
    """
    Settings for controlling how to assign partial charges,
    including the partial charge assignment method, and the
    number of conformers used to generate the partial charges.
    """
