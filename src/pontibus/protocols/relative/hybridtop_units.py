# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
ProtocolUnit implementations for the HybridTopProtocol.
"""

import logging
import os
import pathlib
import warnings
from itertools import chain
from typing import Any

import mdtraj
import numpy as np
import openmmtools
from gufe import SmallMoleculeComponent, SolventComponent
from gufe.settings import ThermoSettings
from gufe.vendor.openff.models.types import ArrayQuantity
from openfe.protocols.openmm_rfe import _rfe_utils
from openfe.protocols.openmm_rfe.equil_rfe_methods import (
    RelativeHybridTopologyProtocolUnit,
    _get_alchemical_charge_difference,
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import (
    AlchemicalSettings,
    LambdaSettings,
)
from openfe.protocols.openmm_utils import (
    multistate_analysis,
    omm_compute,
    settings_validation,
    system_validation,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
    IntegratorSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
)
from openfe.utils import without_oechem_backend
from openff.interchange import Interchange
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import Molecule as OFFMolecule
from openff.units import Quantity, unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import CMMotionRemover, MonteCarloBarostat, System
from openmm import unit as omm_unit
from openmm.app import Topology
from openmmtools import multistate

from pontibus.protocols.relative.settings import HybridTopProtocolSettings
from pontibus.protocols.solvation.base import _get_and_charge_solvent_offmol
from pontibus.utils.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.system_creation import (
    _get_force_field,
    interchange_packmol_creation,
)
from pontibus.utils.system_manipulation import (
    adjust_system,
    copy_interchange_with_replacement,
)

logger = logging.getLogger(__name__)


class HybridTopProtocolUnit(RelativeHybridTopologyProtocolUnit):
    @staticmethod
    def _check_position_overlap(
        mapping: dict[str, dict[int, int]],
        positionsA: ArrayQuantity,
        positionsB: ArrayQuantity,
        threshold: Quantity = 1.0 * unit.angstrom,
    ):
        """
        Sanity check the overlap in positions.

        Parameters
        ----------
        mapping : dict[str, dict[int, int]]
          The system mappings between the two sets of positions.
        positionsA : openff.units.Quantity
          The system A positions.
        positionsB : openff.units.Quantity
          The system B positions.
        tolerance : openff.units.Quantity
          The maximum deivation allowed before an error or warning is raised.

        Raises
        ------
        ValueError
          If any env atoms deviate by more than the threshold.
        UserWarning
          If any core atoms deviate by more than the threshold.
        """
        # Check env mappings
        for key, val in mapping["old_to_new_env_atom_map"].items():
            if np.any(np.abs(positionsB[val] - positionsA[key]) > threshold):
                msg = f"env mapping {key} : {val} deviates by more than {threshold}"
                raise ValueError(msg)

        # Check core mappings
        for key, val in mapping["old_to_new_core_atom_map"].items():
            if np.any(np.abs(positionsB[val] - positionsA[key]) > threshold):
                msg = f"core mapping {key} : {val} deviates by more than {threshold}"
                warnings.warn(msg)
                logging.warning(msg)

    @staticmethod
    def _get_barostat(
        solvent_component: SolventComponent | None,
        thermo_settings: ThermoSettings,
        integrator_settings: IntegratorSettings,
    ) -> MonteCarloBarostat | None:
        """
        Helper to get a barostat for the system.

        Parameters
        ----------
        solvent_component: SolventComponent | None
          The system's SolventComponent, if there is one.
        thermo_settings : ThermoSettings
          The thermodynamic settings.
        integrator_settings : IntegratorSettings
          The integrator settings

        Returns
        -------
        MonteCarloBarostat | None
          None is there is no solvent, a MonteCarloBarostat otherwise.
        """
        if solvent_component is None:
            return None

        return MonteCarloBarostat(
            to_openmm(thermo_settings.pressure),
            to_openmm(thermo_settings.temperature),
            integrator_settings.barostat_frequency.m,
        )

    @staticmethod
    def _get_interchanges(
        small_mols,
        protein_component,
        solvent_component,
        forcefield_settings,
        solvation_settings,
        charge_settings,
    ):
        # Create an smc comp dictionary for stateA
        stateA_smc_comps = dict(chain(small_mols["stateA"], small_mols["both"]))

        # Get solvent offmol if necessary
        if solvent_component is not None:
            solvent_offmol = _get_and_charge_solvent_offmol(
                solvent_component,
                solvation_settings,
                charge_settings,
            )
        else:
            solvent_offmol = None

        # Get the stateA interchange
        with without_oechem_backend():
            interA, comp_residsA = interchange_packmol_creation(
                ffsettings=forcefield_settings,
                solvation_settings=solvation_settings,
                smc_components=stateA_smc_comps,
                protein_component=protein_component,
                solvent_component=solvent_component,
                solvent_offmol=solvent_offmol,
            )

        # Get a list of the charged molecules to create stateB
        stateB_charged_mols = [pair[1] for pair in chain(small_mols["stateB"], small_mols["both"])]
        if solvent_component is not None and solvation_settings.assign_solvent_charges:
            stateB_charged_mols.append(solvent_offmol)

        # Set the stateB interchange
        interB = copy_interchange_with_replacement(
            interchange=interA,
            del_mol=small_mols["stateA"][0][1],
            insert_mol=small_mols["stateB"][0][1],
            force_field=_get_force_field(forcefield_settings),
            charged_molecules=stateB_charged_mols,
        )

        # Fetch the alchemical resids for each state from the comp_resids
        alchem_resids = {
            "stateA": comp_residsA[small_mols["stateA"][0][0]],
            "stateB": np.array([interB.topology.n_molecules - 1], dtype=int),
        }

        return interA, interB, alchem_resids

    def _get_omm_objects(
        self,
        interchange: Interchange,
        forcefield_settings: InterchangeFFSettings,
        thermo_settings: ThermoSettings,
        integrator_settings: IntegratorSettings,
        solvent_component: SolventComponent | None,
    ) -> tuple[Topology, omm_unit.Quantity, System]:
        """
        Helper method to extract OpenMM objects from an Interchange object.

        Parameters
        ----------
        interchange : Interchange
          The Interchange object to get OpenMM objects from.
        forcefield_settings : InterchangeFFSettings
          The force field settings
        thermo_settings : ThermoSettings
          The thermodynamic parameter settings.
        integrator_settings : IntegratorSettings
          The integrator settings.
        solvent_component : SolventComponent | None
          The SolventComponent, if there is one.
        """
        topology = interchange.to_openmm_topology(collate=True)
        positions = to_openmm_positions(
            interchange,
            include_virtual_sites=True,
        )
        system = interchange.to_openmm_system(hydrogen_mass=forcefield_settings.hydrogen_mass)
        adjust_system(
            system=system,
            remove_forces=CMMotionRemover,
            add_forces=self._get_barostat(
                solvent_component=solvent_component,
                thermo_settings=thermo_settings,
                integrator_settings=integrator_settings,
            ),
        )
        return topology, positions, system

    def run(
        self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None
    ) -> dict[str, Any]:
        """Run the relative free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary hybrid
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        scratch_basepath: Pathlike, optional
          Where to store temporary files, defaults to current working directory
        shared_basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.

        Raises
        ------
        error
          Exception if anything failed
        """
        if verbose:
            self.logger.info("Preparing the hybrid topology simulation")
        if scratch_basepath is None:
            scratch_basepath = pathlib.Path(".")
        if shared_basepath is None:
            # use cwd
            shared_basepath = pathlib.Path(".")

        # 0. General setup and settings dependency resolution step

        # Extract relevant settings
        protocol_settings: HybridTopProtocolSettings = self._inputs["protocol"].settings
        stateA = self._inputs["stateA"]
        stateB = self._inputs["stateB"]
        mapping = self._inputs["ligandmapping"]

        forcefield_settings: InterchangeFFSettings = protocol_settings.forcefield_settings
        thermo_settings: ThermoSettings = protocol_settings.thermo_settings
        alchem_settings: AlchemicalSettings = protocol_settings.alchemical_settings
        lambda_settings: LambdaSettings = protocol_settings.lambda_settings
        charge_settings: BasePartialChargeSettings = protocol_settings.partial_charge_settings
        solvation_settings: PackmolSolvationSettings = protocol_settings.solvation_settings
        sampler_settings: MultiStateSimulationSettings = protocol_settings.simulation_settings
        output_settings: MultiStateOutputSettings = protocol_settings.output_settings
        integrator_settings: IntegratorSettings = protocol_settings.integrator_settings

        # is the timestep good for the mass?
        settings_validation.validate_timestep(
            forcefield_settings.hydrogen_mass, integrator_settings.timestep
        )
        # TODO: Also validate various conversions?
        # Convert various time based inputs to steps/iterations
        steps_per_iteration = settings_validation.convert_steps_per_iteration(
            simulation_settings=sampler_settings,
            integrator_settings=integrator_settings,
        )

        equil_steps = settings_validation.get_simsteps(
            sim_length=sampler_settings.equilibration_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=sampler_settings.production_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)
        # We already do this in the Protocol but check the number of alchemical comps
        for state in alchem_comps.values():
            assert len(state) == 1, "too many alchemical components found in one state"

        # Get the change difference between the end states
        # and check if the charge correction used is appropriate
        charge_difference = _get_alchemical_charge_difference(
            mapping,
            forcefield_settings.nonbonded_method,
            alchem_settings.explicit_charge_correction,
            solvent_comp,
        )

        # 1. Create stateA system
        self.logger.info("Parameterizing molecules")

        # a. create (SMC, offmol) dictionaries and assign partial charges
        # calculate partial charges manually if not already given
        # convert to OpenFF here,
        # and keep the molecule around to maintain the partial charges
        off_small_mols: dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]]
        off_small_mols = {
            "stateA": [(alchem_comps["stateA"][0], alchem_comps["stateA"][0].to_openff())],
            "stateB": [(alchem_comps["stateB"][0], alchem_comps["stateB"][0].to_openff())],
            "both": [
                (m, m.to_openff())
                for m in small_mols
                if (m != alchem_comps["stateA"][0] and m != alchem_comps["stateB"][0])
            ],
        }

        self._assign_partial_charges(charge_settings, off_small_mols)

        # Get stateA and stateB interchanges
        stateA_interchange, stateB_interchange, alchem_resids = self._get_interchanges(
            off_small_mols,
            protein_comp,
            solvent_comp,
            forcefield_settings,
            solvation_settings,
            charge_settings,
        )

        # get topology & positions
        stateA_topology, stateA_positions, stateA_system = self._get_omm_objects(
            interchange=stateA_interchange,
            forcefield_settings=forcefield_settings,
            thermo_settings=thermo_settings,
            integrator_settings=integrator_settings,
            solvent_component=solvent_comp,
        )
        stateB_topology, stateB_positions, stateB_system = self._get_omm_objects(
            interchange=stateB_interchange,
            forcefield_settings=forcefield_settings,
            thermo_settings=thermo_settings,
            integrator_settings=integrator_settings,
            solvent_component=solvent_comp,
        )

        #  c. Define correspondence mappings between the two systems
        ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system,
            stateA_topology,
            alchem_resids["stateA"],
            stateB_system,
            stateB_topology,
            alchem_resids["stateB"],
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        # Sanity check the mappings looking at position overlaps
        self._check_position_overlap(
            ligand_mappings,
            from_openmm(stateA_positions),
            from_openmm(stateB_positions),
        )

        # d. if a charge correction is necessary, select alchemical waters
        #    and transform them
        if alchem_settings.explicit_charge_correction:
            alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
                stateA_topology,
                from_openmm(stateA_positions).m,
                charge_difference,
                alchem_settings.explicit_charge_correction_cutoff,
            )
            _rfe_utils.topologyhelpers.handle_alchemical_waters(
                alchem_water_resids,
                stateB_topology,
                stateB_system,
                ligand_mappings,
                charge_difference,
                solvent_comp,
            )

        # 3. Create the hybrid topology
        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
            stateA_system,
            stateA_positions,
            stateA_topology,
            stateB_system,
            stateB_positions,
            stateB_topology,
            old_to_new_atom_map=ligand_mappings["old_to_new_atom_map"],
            old_to_new_core_atom_map=ligand_mappings["old_to_new_core_atom_map"],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=alchem_settings.softcore_LJ.lower() == "gapsys",
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchem_settings.turn_off_core_unique_exceptions,
        )

        # 4. Create lambda schedule
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=lambda_settings.lambda_functions, windows=lambda_settings.lambda_windows
        )

        # pin lambda schedule spacing to n_replicas
        n_replicas = sampler_settings.n_replicas
        if n_replicas != len(lambdas.lambda_schedule):
            errmsg = (
                f"Number of replicas {n_replicas} "
                f"does not equal the number of lambda windows "
                f"{len(lambdas.lambda_schedule)}"
            )
            raise ValueError(errmsg)

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        selection_indices = hybrid_factory.hybrid_topology.select(output_settings.output_indices)

        #  a. Create the multistate reporter
        # convert checkpoint_interval from time to iterations
        chk_intervals = settings_validation.convert_checkpoint_interval_to_iterations(
            checkpoint_interval=output_settings.checkpoint_interval,
            time_per_iteration=sampler_settings.time_per_iteration,
        )

        nc = shared_basepath / output_settings.output_filename
        chk = output_settings.checkpoint_storage_filename

        if output_settings.positions_write_frequency is not None:
            pos_interval = settings_validation.divmod_time_and_check(
                numerator=output_settings.positions_write_frequency,
                denominator=sampler_settings.time_per_iteration,
                numerator_name="output settings' position_write_frequency",
                denominator_name="sampler settings' time_per_iteration",
            )
        else:
            pos_interval = 0

        if output_settings.velocities_write_frequency is not None:
            vel_interval = settings_validation.divmod_time_and_check(
                numerator=output_settings.velocities_write_frequency,
                denominator=sampler_settings.time_per_iteration,
                numerator_name="output settings' velocity_write_frequency",
                denominator_name="sampler settings' time_per_iteration",
            )
        else:
            vel_interval = 0

        reporter = multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=chk_intervals,
            checkpoint_storage=chk,
            position_interval=pos_interval,
            velocity_interval=vel_interval,
        )

        #  b. Write out a PDB containing the subsampled hybrid state
        bfactors = np.zeros_like(selection_indices, dtype=float)  # solvent
        bfactors[
            np.in1d(selection_indices, list(hybrid_factory._atom_classes["unique_old_atoms"]))
        ] = 0.25  # lig A
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes["core_atoms"]))] = (
            0.50  # core
        )
        bfactors[
            np.in1d(selection_indices, list(hybrid_factory._atom_classes["unique_new_atoms"]))
        ] = 0.75  # lig B

        if len(selection_indices) > 0:
            traj = mdtraj.Trajectory(
                hybrid_factory.hybrid_positions[selection_indices, :],
                hybrid_factory.hybrid_topology.subset(selection_indices),
            ).save_pdb(
                shared_basepath / output_settings.output_structure,
                bfactors=bfactors,
            )

        # 10. Get compute platform
        # restrict to a single CPU if running vacuum
        restrict_cpu = forcefield_settings.nonbonded_method.lower() == "nocutoff"
        platform = omm_compute.get_openmm_platform(
            platform_name=protocol_settings.engine_settings.compute_platform,
            gpu_device_index=protocol_settings.engine_settings.gpu_device_index,
            restrict_cpu_count=restrict_cpu,
        )

        # 11. Set the integrator
        # a. Validate integrator settings for current system
        # Virtual sites sanity check - ensure we restart velocities when
        # there are virtual sites in the system
        if hybrid_factory.has_virtual_sites:
            if not integrator_settings.reassign_velocities:
                errmsg = (
                    "Simulations with virtual sites without velocity "
                    "reassignments are unstable in openmmtools"
                )
                raise ValueError(errmsg)

        #  b. create langevin integrator
        integrator = openmmtools.mcmc.LangevinDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.langevin_collision_rate),
            n_steps=steps_per_iteration,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        # 12. Create sampler
        self.logger.info("Creating and setting up the sampler")
        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=sampler_settings,
        )
        # convert early_termination_target_error from kcal/mol to kT
        early_termination_target_error = (
            settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
                thermo_settings.temperature,
                sampler_settings.early_termination_target_error,
            )
        )

        if sampler_settings.sampler_method.lower() == "repex":
            sampler = _rfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = _rfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_minimum_iterations=rta_min_its,
                flatness_criteria=sampler_settings.sams_flatness_criteria,
                gamma0=sampler_settings.sams_gamma0,
            )
        elif sampler_settings.sampler_method.lower() == "independent":
            sampler = _rfe_utils.multistate.HybridMultiStateSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )

        else:
            raise AttributeError(f"Unknown sampler {sampler_settings.sampler_method}")

        sampler.setup(
            n_replicas=sampler_settings.n_replicas,
            reporter=reporter,
            lambda_protocol=lambdas,
            temperature=to_openmm(thermo_settings.temperature),
            endstates=alchem_settings.endstate_dispersion_correction,
            minimization_platform=platform.getName(),
        )

        try:
            # Create context caches (energy + sampler)
            energy_context_cache = openmmtools.cache.ContextCache(
                capacity=None,
                time_to_live=None,
                platform=platform,
            )

            sampler_context_cache = openmmtools.cache.ContextCache(
                capacity=None,
                time_to_live=None,
                platform=platform,
            )

            sampler.energy_context_cache = energy_context_cache
            sampler.sampler_context_cache = sampler_context_cache

            if not dry:  # pragma: no-cover
                # minimize
                if verbose:
                    self.logger.info("Running minimization")

                sampler.minimize(max_iterations=sampler_settings.minimization_steps)

                # equilibrate
                if verbose:
                    self.logger.info("Running equilibration phase")

                sampler.equilibrate(int(equil_steps / steps_per_iteration))

                # production
                if verbose:
                    self.logger.info("Running production phase")

                sampler.extend(int(prod_steps / steps_per_iteration))

                self.logger.info("Production phase complete")

                self.logger.info("Post-simulation analysis of results")
                # calculate relevant analyses of the free energies & sampling
                # First close & reload the reporter to avoid netcdf clashes
                analyzer = multistate_analysis.MultistateEquilFEAnalysis(
                    reporter,
                    sampling_method=sampler_settings.sampler_method.lower(),
                    result_units=unit.kilocalorie_per_mole,
                )
                analyzer.plot(filepath=shared_basepath, filename_prefix="")
                analyzer.close()

            else:
                # clean up the reporter file
                fns = [
                    shared_basepath / output_settings.output_filename,
                    shared_basepath / output_settings.checkpoint_storage_filename,
                ]
                for fn in fns:
                    os.remove(fn)
        finally:
            # close reporter when you're done, prevent
            # file handle clashes
            reporter.close()

            # clear GPU contexts
            # TODO: use cache.empty() calls when openmmtools #690 is resolved
            # replace with above
            for context in list(energy_context_cache._lru._data.keys()):
                del energy_context_cache._lru._data[context]
            for context in list(sampler_context_cache._lru._data.keys()):
                del sampler_context_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_context_cache, energy_context_cache

            if not dry:
                del integrator, sampler

        if not dry:  # pragma: no-cover
            return {"nc": nc, "last_checkpoint": chk, **analyzer.unit_results_dict}
        else:
            return {"debug": {"sampler": sampler}}
