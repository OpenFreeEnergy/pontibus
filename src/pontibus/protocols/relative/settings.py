# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium RFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run hybrid topology
relative free energies using OpenMM.

See Also
--------
openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol
"""
from typing import Literal, Optional

from gufe.settings import BaseForceFieldSettings, ThermoSettings
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)
from openff.models.types import FloatQuantity, ArrayQuantity
from openff.units import unit
from openff.interchange.components._packmol import _box_vectors_are_in_reduced_form
from pydantic.v1 import validator, root_validator
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)


class HybridTopologyProtocolSettings(RelativeHybridTopologyProtocolSettings):
    """
    Configuration object for ``HybridTopologyRFEProtocol``.

    See Also
    --------
    pontibus.protocols.relative.HybridTopologyRFEProtocol
    """

    # Inherited things
    forcefield_settings: InterchangeFFSettings
    """Parameters to set up in the force field"""

    solvation_settings: PackmolSolvationSettings
    """Settings for solvating the system."""
