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

import uuid

from gufe import ChemicalSystem, ComponentMapping, ProtocolDAGResult, ProtocolUnit
from gufe.settings import ThermoSettings
from openfe.protocols.openmm_rfe.equil_rfe_methods import (
    RelativeHybridTopologyProtocol,
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    AlchemicalSettings,
    LambdaSettings,
)
from openfe.protocols.openmm_rfe.hybridtop_units import (
    HybridTopologyMultiStateAnalysisUnit,
    HybridTopologyMultiStateSimulationUnit,
)
from openfe.protocols.openmm_utils import settings_validation, system_validation
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
)
from openff.units import unit

from pontibus.protocols.relative.hybridtop_protocol_results import HybridTopProtocolResult
from pontibus.protocols.relative.hybridtop_units import HybridTopProtocolSetupUnit
from pontibus.protocols.relative.settings import HybridTopProtocolSettings
from pontibus.utils.settings import (
    InterchangeFFSettings,
    InterchangeOpenMMSolvationSettings,
)


class HybridTopProtocol(RelativeHybridTopologyProtocol):
    """
    Relative Free Energy calculations using OpenMM and OpenMMTools.

    Based on `Perses <https://github.com/choderalab/perses>`_

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol`
    :class:`pontibus.protocols.relative.HybridTopSettings`
    :class:`pontibus.protocols.relative.HybridTopResult`
    :class:`pontibus.protocols.relative.HybridTopProtocolSetupUnit`
    """

    result_cls = HybridTopProtocolResult
    _settings_cls = HybridTopProtocolSettings
    _settings: HybridTopProtocolSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol
        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.
        Returns
        -------
        Settings
          a set of default settings
        """
        return HybridTopProtocolSettings(
            protocol_repeats=3,
            forcefield_settings=InterchangeFFSettings(),
            thermo_settings=ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=InterchangeOpenMMSolvationSettings(
                target_density=0.75 * unit.grams / unit.mL
            ),
            alchemical_settings=AlchemicalSettings(softcore_LJ="gapsys"),
            lambda_settings=LambdaSettings(),
            simulation_settings=MultiStateSimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            ),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            output_settings=MultiStateOutputSettings(),
        )

    def _validate(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None,
        extends: ProtocolDAGResult | None = None,
    ) -> None:
        # Check we're not trying to extend
        if extends:
            raise ValueError("Can't extend simulations yet")

        # Validate the end states
        self._validate_endstates(stateA, stateB)

        # Validate the mapping
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)
        self._validate_mapping(mapping, alchem_comps)

        # Validate the small molecule components
        self._validate_smcs(stateA, stateB)

        # Validate solvent component
        nonbond = self.settings.forcefield_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # Validate integrator things
        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None,
        extends: ProtocolDAGResult | None,
    ) -> list[ProtocolUnit]:
        # Run validation
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # Get alchemical components & mapping
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)
        ligandmapping = mapping[0] if isinstance(mapping, list) else mapping

        Anames = ",".join(c.name for c in alchem_comps["stateA"])
        Bnames = ",".join(c.name for c in alchem_comps["stateB"])
        n_repeats = self.settings.protocol_repeats

        setup_units = []
        simulation_units = []
        analysis_units = []

        for i in range(n_repeats):
            repeat_id = int(uuid.uuid4())

            setup = HybridTopProtocolSetupUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                ligandmapping=ligandmapping,
                alchemical_components=alchem_comps,
                generation=0,
                repeat_id=repeat_id,
                name=f"HybridTop Setup: {Anames} to {Bnames} repeat {i} generation 0",
            )

            simulation = HybridTopologyMultiStateSimulationUnit(
                protocol=self,
                setup_results=setup,
                generation=0,
                repeat_id=repeat_id,
                name=f"HybridTop Simulation: {Anames} to {Bnames} repeat {i} generation 0",
            )

            analysis = HybridTopologyMultiStateAnalysisUnit(
                protocol=self,
                setup_results=setup,
                simulation_results=simulation,
                generation=0,
                repeat_id=repeat_id,
                name=f"HybridTop Analysis: {Anames} to {Bnames} repeat {i} generation 0",
            )

            setup_units.append(setup)
            simulation_units.append(simulation)
            analysis_units.append(analysis)

        return [*setup_units, *simulation_units, *analysis_units]
