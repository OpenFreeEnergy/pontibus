# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings classes for RFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run the following:
  * HybridTopProtocol

See Also
--------
pontibus.protocols.relative.HybridTopProtocol
"""
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
)
from pontibus.utils.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)


class HybridTopProtocolSettings(RelativeHybridTopologyProtocolSettings):
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
