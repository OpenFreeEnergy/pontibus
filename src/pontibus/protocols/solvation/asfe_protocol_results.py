# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.protocols.openmm_afe import AbsoluteSolvationProtocolResult


class ASFEProtocolResult(AbsoluteSolvationProtocolResult):
    """
    Results class for the ASFEProtocol.

    Notes
    -----
    * Derives from OpenFE's AbsoluteSolvationProtocolResult with the intent
      of extending further if necessary in the future.
    """
