# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for ASFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute
solvation free energies using OpenMM.

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
from pontibus.utils.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)


class ExperimentalAlchemicalSettings(AlchemicalSettings):
    experimental: bool = False
    """
    Enable the use of experimental alchemical features.

    This includes, but is not limited to, support for virtual sites
    in alchemical transformations.
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
