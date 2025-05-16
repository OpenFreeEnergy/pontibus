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
This Protocol is based on, and leverages components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""
from __future__ import annotations

import os
import logging
from collections import defaultdict
import uuid
import warnings
import json
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from openff.units import unit
from openff.units.openmm import to_openmm, from_openmm, ensure_quantity
from openff.toolkit.topology import Molecule as OFFMolecule
from openmmtools import multistate
from typing import Optional
import pathlib
from typing import Any, Iterable, Union
import openmmtools
import mdtraj
import subprocess
from rdkit import Chem

import gufe
from gufe import (
    settings, ChemicalSystem, LigandAtomMapping, Component, ComponentMapping,
    SmallMoleculeComponent, SolventComponent,
)

from .equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
    OpenMMSolvationSettings, AlchemicalSettings, LambdaSettings,
    MultiStateSimulationSettings, OpenMMEngineSettings,
    IntegratorSettings, MultiStateOutputSettings,
    OpenFFPartialChargeSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
)
from ..openmm_utils import (
    system_validation, settings_validation, system_creation,
    multistate_analysis, charge_generation, omm_compute,
)
from openfe.protocols.openmm_rfe.equil_rfe_methods import(
    _get_resname,
    _get_alchemical_charge_difference,
    _validate_alchemical_components,
    RelativeHybridTopologyProtocolResults,
    RelativeHybridTopologyProtocol,
)
from . import _rfe_utils
from ...utils import without_oechem_backend, log_system_probe
from ...analysis import plotting
from openfe.due import due, Doi


logger = logging.getLogger(__name__)


class HybridTopologyProtocolResult(RelativeHybridTopologyProtocolResult):
    """
    Results class for the HybridTopologyProtocol class.

    Inherits from
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocolResult`.
    """


class HybridTopologyProtocol(RelativeHybridTopologyProtocol):
    """
    Relative Free Energy calculations using OpenMM and OpenMMTools.

    Based on `Perses <https://github.com/choderalab/perses>`_

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol`
    """
    result_cls = HybridTopologyProtocolResult
    _settings_cls = HybridTopologyProtocolSettings
    _settings: HybridTopologyProtocolSettings

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
        return HybridTopologyProtocolSettings(
            protocol_repeats=3,
            forcefield_settings=InterchangeFFSettings(),
            thermo_settings=ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=PackmolSolvationSettings(),
            alchemical_settings=AlchemicalSettings(softcore_LJ='gapsys'),
            lambda_settings=LambdaSettings(),
            simulation_settings=MultiStateSimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            ),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            output_settings=MultiStateOutputSettings(),
        )

    @staticmethod
    def _get_interchanges(
        small_mols,
        protein_comp,
        solvent_comp,
        forcefield_settings,
        solvation_settings,
    ):
        stateA_smc_comps = dict(
            chain(
                small_mols['stateA'],
                small_mols['both'],
            )
        )
        interchangeA, comp_residsA = interchange_packmol_creation(
            ffsettings=forcefield_settings,
            solvation_settings=solvation_settings,
            smc_components=stateA_smc_comps,
        )
        interchangeB = interchangeA.copy()
        # We assume a single alchemical component, which this Protocol does
        # anyways
        molA = small_mols['stateA'][0][1]
        molB = small_mols['stateB'][0][1]

        molA_idx = None
        for idx, mol in enumerate(interchangeB.topology._molecules):
            if mol.is_isomoprhic_with(molA):
                molA_idx = idx

        if molA_idx is not None:
            interchangeB.topology._molecules.pop(remove_idx)
        else:
            raise ValueError("Couldn't find an alchemical molecule to remove in stateA")


        for atom in molB.atoms:
            atom.metadata['residue_number'] = comp_resids[small_mols['stateA']]
        interchangeB.topology.add_molecule(molB)

        alchem_resids = {
            "stateA": molA_idx,
            "stateB": len(interchange.topology._molecules)-1
        }

        return interchangeA, interchangeB, 
        

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]],
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: Extensions?
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        # Get alchemical components & validate them + mapping
        alchem_comps = system_validation.get_alchemical_components(
            stateA, stateB
        )
        _validate_alchemical_components(alchem_comps, mapping)
        ligandmapping = mapping[0] if isinstance(mapping, list) else mapping  # type: ignore

        # Validate solvent component
        nonbond = self.settings.forcefield_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(
            self.settings.solvation_settings
        )

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
        Anames = ','.join(c.name for c in alchem_comps['stateA'])
        Bnames = ','.join(c.name for c in alchem_comps['stateB'])
        # our DAG has no dependencies, so just list units
        n_repeats = self.settings.protocol_repeats
        units = [HybridTopologyProtocolUnit(
            protocol=self,
            stateA=stateA, stateB=stateB,
            ligandmapping=ligandmapping,  # type: ignore
            generation=0, repeat_id=int(uuid.uuid4()),
            name=f'{Anames} to {Bnames} repeat {i} generation 0')
            for i in range(n_repeats)]

        return units


class HybridTopologyProtocolUnit(RelativeHybridTopologyProtocolUnit):
    """
    Calculates the relative free energy of an alchemical ligand transformation.
    """

    def run(self, *, dry=False, verbose=True,
            scratch_basepath=None,
            shared_basepath=None) -> dict[str, Any]:
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
            scratch_basepath = pathlib.Path('.')
        if shared_basepath is None:
            # use cwd
            shared_basepath = pathlib.Path('.')

        # 0. General setup and settings dependency resolution step

        # Extract relevant settings
        protocol_settings: HybridTopologyProtocolSettings = self._inputs['protocol'].settings
        stateA = self._inputs['stateA']
        stateB = self._inputs['stateB']  # TODO: open an issue about this not being used.
        mapping = self._inputs['ligandmapping']

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
            forcefield_settings.hydrogen_mass,
            integrator_settings.timestep
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

        # a. create offmol dictionaries and assign partial charges
        # calculate partial charges manually if not already given
        # and keep the off molecule around to maintain the partial charges
        off_small_mols: dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]]
        off_small_mols = {
            'stateA': [(mapping.componentA, mapping.componentA.to_openff())],
            'stateB': [(mapping.componentB, mapping.componentB.to_openff())],
            'both': [(m, m.to_openff()) for m in small_mols
                     if (m != mapping.componentA and m != mapping.componentB)]
        }

        self._assign_partial_charges(charge_settings, off_small_mols)

        stateA_interchange, stateB_interchange, alchem_resids = self._get_interchanges(
            off_small_mols,
            protein_comp,
            solvent_comp,
            forcefield_settings,
            solvation_settings,
        )

        stateA_topology = stateA_interchange.to_openmm_topology(collate=True)
        stateA_positions = to_openmm_positions(
            stateA_interchange,
            include_virtual_sites=True,
        )
        stateA_system = stateA_interchange.to_openmm_system(
            hydrogen_mass=forcefield_settings.hydrogen_mass
        )
        stateB_topology = stateB_interchange.to_openmm_topology(collate=True)
        stateB_positions = to_openmm_positions(
            stateB_interchange,
            include_virtual_sites=True,
        )
        stateB_system = stateB_interchange.to_openmm_system(
            hydrogen_mass=forcefield_settings.hydrogen_mass
        )

        #  c. Define correspondence mappings between the two systems
        ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology, alchem_resids['stateA'],
            stateB_system, stateB_topology, alchem_resids['stateB'],
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        # d. if a charge correction is necessary, select alchemical waters
        #    and transform them
        if alchem_settings.explicit_charge_correction:
            alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
                stateA_topology, stateA_positions,
                charge_difference,
                alchem_settings.explicit_charge_correction_cutoff,
            )
            _rfe_utils.topologyhelpers.handle_alchemical_waters(
                alchem_water_resids, stateB_topology, stateB_system,
                ligand_mappings, charge_difference,
                solvent_comp,
            )

        #  e. Finally get the positions
        stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings, stateA_topology, stateB_topology,
            old_positions=ensure_quantity(stateA_positions, 'openmm'),
            insert_positions=ensure_quantity(off_small_mols['stateB'][0][1].conformers[0], 'openmm'),
        )

        # 3. Create the hybrid topology
        # a. Get softcore potential settings
        if alchem_settings.softcore_LJ.lower() == 'gapsys':
            softcore_LJ_v2 = True
        elif alchem_settings.softcore_LJ.lower() == 'beutler':
            softcore_LJ_v2 = False
        # b. Get hybrid topology factory
        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
            stateA_system, stateA_positions, stateA_topology,
            stateB_system, stateB_positions, stateB_topology,
            old_to_new_atom_map=ligand_mappings['old_to_new_atom_map'],
            old_to_new_core_atom_map=ligand_mappings['old_to_new_core_atom_map'],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchem_settings.turn_off_core_unique_exceptions,
        )

        # 4. Create lambda schedule
        # TODO - this should be exposed to users, maybe we should offer the
        # ability to print the schedule directly in settings?
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=lambda_settings.lambda_functions,
            windows=lambda_settings.lambda_windows
        )

        # PR #125 temporarily pin lambda schedule spacing to n_replicas
        n_replicas = sampler_settings.n_replicas
        if n_replicas != len(lambdas.lambda_schedule):
            errmsg = (f"Number of replicas {n_replicas} "
                      f"does not equal the number of lambda windows "
                      f"{len(lambdas.lambda_schedule)}")
            raise ValueError(errmsg)

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        selection_indices = hybrid_factory.hybrid_topology.select(
                output_settings.output_indices
        )

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
                denominator_name="sampler settings' time_per_iteration"
            )
        else:
            pos_interval = 0

        if output_settings.velocities_write_frequency is not None:
            vel_interval = settings_validation.divmod_time_and_check(
                numerator=output_settings.velocities_write_frequency,
                denominator=sampler_settings.time_per_iteration,
                numerator_name="output settings' velocity_write_frequency",
                denominator_name="sampler settings' time_per_iteration"
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
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_old_atoms']))] = 0.25  # lig A
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['core_atoms']))] = 0.50  # core
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_new_atoms']))] = 0.75  # lig B
        # bfactors[np.in1d(selection_indices, protein)] = 1.0  # prot+cofactor

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
        restrict_cpu = forcefield_settings.nonbonded_method.lower() == 'nocutoff'
        platform = omm_compute.get_openmm_platform(
            platform_name=protocol_settings.engine_settings.compute_platform,
            gpu_device_index=protocol_settings.engine_settings.gpu_device_index,
            restrict_cpu_count=restrict_cpu
        )

        # 11. Set the integrator
        # a. Validate integrator settings for current system
        # Virtual sites sanity check - ensure we restart velocities when
        # there are virtual sites in the system
        if hybrid_factory.has_virtual_sites:
            if not integrator_settings.reassign_velocities:
                errmsg = ("Simulations with virtual sites without velocity "
                          "reassignments are unstable in openmmtools")
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
        early_termination_target_error = settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
            thermo_settings.temperature,
            sampler_settings.early_termination_target_error,
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
        elif sampler_settings.sampler_method.lower() == 'independent':
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
                capacity=None, time_to_live=None, platform=platform,
            )

            sampler_context_cache = openmmtools.cache.ContextCache(
                capacity=None, time_to_live=None, platform=platform,
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

                sampler.equilibrate(
                    int(equil_steps / steps_per_iteration)  # type: ignore
                )

                # production
                if verbose:
                    self.logger.info("Running production phase")

                sampler.extend(
                    int(prod_steps / steps_per_iteration)  # type: ignore
                )

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
                fns = [shared_basepath / output_settings.output_filename,
                       shared_basepath / output_settings.checkpoint_storage_filename]
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
            for context in list(
                    openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_context_cache, energy_context_cache

            if not dry:
                del integrator, sampler

        if not dry:  # pragma: no-cover
            return {
                'nc': nc,
                'last_checkpoint': chk,
                **analyzer.unit_results_dict
            }
        else:
            return {'debug': {'sampler': sampler}}

    @staticmethod
    def structural_analysis(scratch, shared) -> dict:
        # don't put energy analysis in here, it uses the open file reporter
        # whereas structural stuff requires that the file handle is closed
        # TODO: we should just make openfe_analysis write an npz instead!
        analysis_out = scratch / 'structural_analysis.json'

        ret = subprocess.run(
            [
                'openfe_analysis',  # CLI entry point
                'RFE_analysis',  # CLI option
                str(shared),  # Where the simulation.nc fille
                str(analysis_out)  # Where the analysis json file is written
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if ret.returncode:
            return {'structural_analysis_error': ret.stderr}

        with open(analysis_out, 'rb') as f:
            data = json.load(f)

        savedir = pathlib.Path(shared)
        if d := data['protein_2D_RMSD']:
            fig = plotting.plot_2D_rmsd(d)
            fig.savefig(savedir / "protein_2D_RMSD.png")
            plt.close(fig)
            f2 = plotting.plot_ligand_COM_drift(data['time(ps)'], data['ligand_wander'])
            f2.savefig(savedir / "ligand_COM_drift.png")
            plt.close(f2)

        f3 = plotting.plot_ligand_RMSD(data['time(ps)'], data['ligand_RMSD'])
        f3.savefig(savedir / "ligand_RMSD.png")
        plt.close(f3)

        # Save to numpy compressed format (~ 6x more space efficient than JSON)
        np.savez_compressed(
            shared / "structural_analysis.npz",
            protein_RMSD=np.asarray(
                data["protein_RMSD"], dtype=np.float32
            ),
            ligand_RMSD=np.asarray(
                data["ligand_RMSD"], dtype=np.float32
            ),
            ligand_COM_drift=np.asarray(
                data["ligand_wander"], dtype=np.float32
            ),
            protein_2D_RMSD=np.asarray(
                data["protein_2D_RMSD"], dtype=np.float32
            ),
            time_ps=np.asarray(
                data["time(ps)"], dtype=np.float32
            ),
        )

        return {'structural_analysis': shared / "structural_analysis.npz"}

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch,
                           shared_basepath=ctx.shared)

        structural_analysis_outputs = self.structural_analysis(
            ctx.scratch, ctx.shared
        )

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs,
            **structural_analysis_outputs,
        }
