# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import uuid
import warnings

import gufe
import numpy as np
from gufe import ChemicalSystem, SolventComponent
from gufe.settings import ThermoSettings
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
from openfe.protocols.openmm_afe.equil_afe_settings import LambdaSettings
from openfe.protocols.openmm_utils import settings_validation, system_validation
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    MDOutputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
)
from openff.units import unit

from pontibus.protocols.solvation.asfe_protocol_results import ASFEProtocolResult
from pontibus.protocols.solvation.asfe_protocol_units import (
    ASFESolventAnalysisUnit,
    ASFESolventSetupUnit,
    ASFESolventSimUnit,
    ASFEVacuumAnalysisUnit,
    ASFEVacuumSetupUnit,
    ASFEVacuumSimUnit,
)
from pontibus.protocols.solvation.settings import (
    ASFESettings,
    ExperimentalAlchemicalSettings,
    InterchangeFFSettings,
    PackmolSolvationSettings,
)


class ASFEProtocol(AbsoluteSolvationProtocol):
    result_cls = ASFEProtocolResult
    _settings_cls = ASFESettings
    _settings: ASFESettings

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
        return ASFESettings(
            protocol_repeats=3,
            solvent_forcefield_settings=InterchangeFFSettings(
                nonbonded_method="pme",
            ),
            vacuum_forcefield_settings=InterchangeFFSettings(
                nonbonded_method="nocutoff",
            ),
            thermo_settings=ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            alchemical_settings=ExperimentalAlchemicalSettings(),
            lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0, 0.25, 0.5, 0.75, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                ],
                lambda_vdw=[
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0,
                ],
                lambda_restraints=[
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=PackmolSolvationSettings(),
            vacuum_engine_settings=OpenMMEngineSettings(),
            solvent_engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.5 * unit.nanosecond,
                equilibration_length=0.5 * unit.nanosecond,
                production_length=9.5 * unit.nanosecond,
            ),
            solvent_equil_output_settings=MDOutputSettings(
                equil_nvt_structure="equil_nvt_structure.pdb",
                equil_npt_structure="equil_npt_structure.pdb",
                production_trajectory_filename="production_equil.xtc",
                log_output="equil_simulation.log",
            ),
            solvent_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=1.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_filename="solvent.nc",
                checkpoint_storage_filename="solvent_checkpoint.nc",
            ),
            vacuum_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=None,
                equilibration_length=0.2 * unit.nanosecond,
                production_length=0.5 * unit.nanosecond,
            ),
            vacuum_equil_output_settings=MDOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure="equil_structure.pdb",
                production_trajectory_filename="production_equil.xtc",
                log_output="equil_simulation.log",
            ),
            vacuum_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=0.5 * unit.nanosecond,
                production_length=2.0 * unit.nanosecond,
            ),
            vacuum_output_settings=MultiStateOutputSettings(
                output_filename="vacuum.nc",
                checkpoint_storage_filename="vacuum_checkpoint.nc",
            ),
        )  # fmt: skip

    @staticmethod
    def _validate_solvent(state: ChemicalSystem, nonbonded_method: str):
        """
        Checks that the ChemicalSystem component has the right solvent
        composition for an input nonbonded_methtod.

        Parameters
        ----------
        state : ChemicalSystem
          The chemical system to inspect.
        nonbonded_method : str
          The nonbonded method to be applied for the simulation.

        Raises
        ------
        ValueError
          * If there are multiple SolventComponents in the ChemicalSystem.
          * If there is a SolventComponent and the `nonbonded_method` is
            `nocutoff`.
        """
        solv = state.get_components_of_type(SolventComponent)

        if len(solv) > 0 and nonbonded_method.lower() == "nocutoff":
            errmsg = "nocutoff cannot be used for solvent transformations"
            raise ValueError(errmsg)

        if len(solv) == 0 and nonbonded_method.lower() == "pme":
            errmsg = "PME cannot be used for vacuum transform"
            raise ValueError(errmsg)

        if len(solv) > 1:
            errmsg = "Multiple SolventComponent found, only one is supported"
            raise ValueError(errmsg)

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ):
        # Check we're not extending
        if extends is not None:
            # This should be a NotImplementedError, but the underlying
            # `validate` method wraps a call to `_validate` around a
            # NotImplementedError exception guard
            raise ValueError("Can't extend simulations yet")

        # Check we're not using a mapping, since we're not doing anything with it
        if mapping is not None:
            wmsg = "A mapping was passed but is not used by this Protocol."
            warnings.warn(wmsg)

        # Validate the endstates & alchemical components
        self._validate_endstates(stateA, stateB)

        # Validate the lambda schedule
        for solv_sets in (
            self.settings.solvent_simulation_settings,
            self.settings.vacuum_simulation_settings,
        ):
            self._validate_lambda_schedule(
                self.settings.lambda_settings,
                solv_sets,
            )

        # Check nonbond & solvent compatibility
        solv_nonbonded_method = self.settings.solvent_forcefield_settings.nonbonded_method
        vac_nonbonded_method = self.settings.vacuum_forcefield_settings.nonbonded_method

        # Use the more complete system validation solvent checks
        # Note: we have to use a special version for arbitrary solvent for now
        self._validate_solvent(stateA, solv_nonbonded_method)

        # Gas phase is always gas phase
        if vac_nonbonded_method.lower() != "nocutoff":
            errmsg = (
                "Only the nocutoff nonbonded_method is supported for "
                f"vacuum calculations, {vac_nonbonded_method} was "
                "passed"
            )
            raise ValueError(errmsg)

        # Check vacuum equilibration MD settings is 0 ns
        nvt_time = self.settings.vacuum_equil_simulation_settings.equilibration_length_nvt
        if nvt_time is not None:
            if not np.allclose(nvt_time, 0 * unit.nanosecond):
                errmsg = "NVT equilibration cannot be run in vacuum simulation"
                raise ValueError(errmsg)

        # Validate integrator things
        settings_validation.validate_timestep(
            self.settings.vacuum_forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

        settings_validation.validate_timestep(
            self.settings.solvent_forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ) -> list[gufe.ProtocolUnit]:
        # Run validation
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # Get the alchemical components and the name of the alchemical smc
        alchem_comps = system_validation.get_alchemical_components(
            stateA,
            stateB,
        )

        alchname = alchem_comps["stateA"][0].name

        unit_classes = {
            "solvent": {
                "setup": ASFESolventSetupUnit,
                "simulation": ASFESolventSimUnit,
                "analysis": ASFESolventAnalysisUnit,
            },
            "vacuum": {
                "setup": ASFEVacuumSetupUnit,
                "simulation": ASFEVacuumSimUnit,
                "analysis": ASFEVacuumAnalysisUnit,
            },
        }

        protocol_units: dict[str, list[gufe.ProtocolUnit]] = {"solvent": [], "vacuum": []}

        for phase in ["solvent", "vacuum"]:
            for i in range(self.settings.protocol_repeats):
                repeat_id = int(uuid.uuid4())

                setup = unit_classes[phase]["setup"](
                    protocol=self,
                    stateA=stateA,
                    stateB=stateB,
                    alchemical_components=alchem_comps,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"ASFE Setup: {alchname} {phase} leg: repeat {i} generation 0",
                )

                simulation = unit_classes[phase]["simulation"](
                    protocol=self,
                    stateA=stateA,
                    alchemical_components=alchem_comps,
                    setup_results=setup,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"ASFE Simulation: {alchname} {phase} leg: repeat {i} generation 0",
                )

                analysis = unit_classes[phase]["analysis"](
                    protocol=self,
                    setup_results=setup,
                    simulation_results=simulation,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"ASFE Analysis: {alchname} {phase} leg: repeat {i} generation 0",
                )

                protocol_units[phase] += [setup, simulation, analysis]

        return protocol_units["solvent"] + protocol_units["vacuum"]
