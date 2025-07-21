# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy methods using OpenMM and OpenMMTools in a
Perses-like manner.
This module implements the necessary methodology toolking to run calculate a
ligand relative free energy transformation using OpenMM tools and one of the
following methods:
    - Hamiltonian Replica Exchange
    - Self-adjusted mixture sampling
    - Independent window sampling

Acknowledgements
----------------
This Protocol is a subclass of the OpenFE RelativeHybridTopologyProtocol.
This Protocol is based on, and leverages components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""

from openfe.protocols.openmm_rfe.equil_rfe_methods import(
    RelativeHybridTopologyProtocolResults,
)


class HybridTopProtocolResult(RelativeHybridTopologyProtocolResult):
    """
    Results class for the HybridTopologyProtocol class.
    Inherits from
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocolResult`.
    """
