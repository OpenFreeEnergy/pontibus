# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import logging
from typing import Any, Optional, Union

import gufe
import numpy.typing as npt
import openmm
import openmmtools
from gufe import ProteinComponent, SmallMoleculeComponent, SolventComponent
from gufe.settings import SettingsBaseModel
from openfe.protocols.openmm_afe.base import BaseAbsoluteUnit
from openfe.protocols.openmm_utils import charge_generation, settings_validation
from openfe.utils import log_system_probe, without_oechem_backend
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import app
from openmmtools import multistate
from openmmtools.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
    AlchemicalState,
)
from openmmtools.states import (
    SamplerState,
    ThermodynamicState,
    create_thermodynamic_state_protocol,
)

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation.settings import (
    IntegratorSettings,
    OpenFFPartialChargeSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.system_creation import interchange_packmol_creation

logger = logging.getLogger(__name__)


class BaseASFEUnit(BaseAbsoluteUnit):

    _simtype: str

    @staticmethod
    def _get_and_charge_solvent_offmol(
        solvent_component: Union[SolventComponent, ExtendedSolventComponent],
        solvation_settings: PackmolSolvationSettings,
        partial_charge_settings: OpenFFPartialChargeSettings,
    ) -> OFFMolecule:
        """
        Helper method to fetch the solvent offmol either
        from an existing solvent_smcs, or from smiles.

        Parameters
        ----------
        solvent_component : SolventComponent
          smiles for the solvent molecule
        solvation_settings : PackmolSolvationSettings
          Settings defining how the system will be solvated
        partial_charge_settings : OpenFFPartialChargeSettigns
          Settings defining how partial charges are applied

        Returns
        -------
        offmol : openff.toolkit.Molecule
        """
        # Get the solvent offmol
        if isinstance(solvent_component, ExtendedSolventComponent):
            solvent_offmol = solvent_component.solvent_molecule.to_openff()
        else:
            solvent_offmol = OFFMolecule.from_smiles(solvent_component.smiles)

        # Assign solvent offmol charges if necessary
        if solvation_settings.assign_solvent_charges:
            if solvent_offmol.n_conformers == 0:
                n_conf = 1
            else:
                n_conf = partial_charge_settings.number_of_conformers

            charge_generation.assign_offmol_partial_charges(
                offmol=solvent_offmol,
                overwrite=False,
                method=partial_charge_settings.partial_charge_method,
                toolkit_backend=partial_charge_settings.off_toolkit_backend,
                generate_n_conformers=n_conf,
                nagl_model=partial_charge_settings.nagl_model,
            )

        return solvent_offmol

    @staticmethod
    def _validate_vsites(
        system: openmm.System, integrator_settings: IntegratorSettings
    ) -> None:
        """
        Validate virtual site handling for alchemical system.

        Parameters
        ----------
        System : openmm.System
          System to validate.
        integrator_settings : IntegratorSettings
          Langevin integrator settings to verify against.

        Returns
        -------
        None

        Notes
        -----
        * Small placeholder for a larger thing.
        """
        has_virtual_sites: bool = False
        for ix in range(system.getNumParticles()):
            if system.isVirtualSite(ix):
                has_virtual_sites = True

        if has_virtual_sites:
            if not integrator_settings.reassign_velocities:
                errmsg = (
                    "Simulations with virtual sites without velocity "
                    "reassignments are unstable"
                )
                raise ValueError(errmsg)

    def _get_omm_objects(
        self,
        settings: dict[str, SettingsBaseModel],
        protein_component: Optional[ProteinComponent],
        solvent_component: Optional[SolventComponent],
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> tuple[
        app.Topology, openmm.System, openmm.unit.Quantity, dict[str, npt.NDArray]
    ]:
        """
        Get the OpenMM Topology, Positions and System of the
        parameterised system.

        Parameters
        ----------
        settings : dict[str, SettingsBaseModel]
          Protocol settings
        protein_component : Optional[ProteinComponent]
          Protein component for the system.
        solvent_component : Optional[SolventComponent]
          Solvent component for the system.
        smc_components : dict[str, OFFMolecule]
          SmallMoleculeComponents defining ligands to be added to the system

        Returns
        -------
        topology : app.Topology
          OpenMM Topology object describing the parameterized system.
        system : openmm.System
          An non-alchemical OpenMM System of the simulated system.
        positions : openmm.unit.Quantity
          Positions of the system.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.

        Notes
        -----
        * For now this method solely calls interchange system creation for
          solvation.
        """
        if self.verbose:
            self.logger.info("Parameterizing system")

        # Set partial charges for all smcs
        self._assign_partial_charges(settings["charge_settings"], smc_components)

        # Get solvent offmol if necessary
        if solvent_component is not None:
            solvent_offmol = self._get_and_charge_solvent_offmol(
                solvent_component,
                settings["solvation_settings"],
                settings["charge_settings"],
            )
        else:
            solvent_offmol = None

        # Create your interchange object
        with without_oechem_backend():
            interchange, comp_resids = interchange_packmol_creation(
                ffsettings=settings["forcefield_settings"],
                solvation_settings=settings["solvation_settings"],
                smc_components=smc_components,
                protein_component=protein_component,
                solvent_component=solvent_component,
                solvent_offmol=solvent_offmol,
            )

        # Get omm objects back
        omm_topology = interchange.to_openmm_topology()
        omm_system = interchange.to_openmm_system(
            hydrogen_mass=settings["forcefield_settings"].hydrogen_mass
        )
        positions = to_openmm_positions(interchange, include_virtual_sites=True)

        # Post creation system validation
        self._validate_vsites(omm_system, settings["integrator_settings"])

        return omm_topology, omm_system, positions, comp_resids

    def run(
        self,
        dry=False,
        verbose=True,
        scratch_basepath=None,
        shared_basepath=None,
    ) -> dict[str, Any]:
        """Run the absolute free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, sampler, etc...) but without
          running the simulation, default False
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging, default True
        scratch_basepath : pathlib.Path
          Path to the scratch (temporary) directory space.
        shared_basepath : pathlib.Path
          Path to the shared (persistent) directory space.

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.
        """

        # 0. Generaly preparation tasks
        self._prepare(verbose, scratch_basepath, shared_basepath)

        # 1. Get components
        alchem_comps, solv_comp, prot_comp, smc_comps = self._get_components()

        # 2. Get settings
        settings = self._handle_settings()

        # 3. Get OpenMM topology, positions and system
        omm_topology, omm_system, positions, comp_resids = self._get_omm_objects(
            settings,
            prot_comp,
            solv_comp,
            smc_comps,
        )

        # 4. Pre-equilbrate System (Test + Avoid NaNs + get stable system)
        positions = self._pre_equilibrate(
            omm_system, omm_topology, positions, settings, dry
        )

        # 5. Get lambdas
        lambdas = self._get_lambda_schedule(settings)

        # 6. Get alchemical system
        if settings["alchemical_settings"].experimental:
            errmsg = "experimental factory code is not yet implemented"
            raise ValueError(errmsg)
        else:
            alchem_factory, alchem_system, alchem_indices = self._get_alchemical_system(
                omm_topology, omm_system, comp_resids, alchem_comps
            )

        # 7. Get compound and sampler states
        sampler_states, cmp_states = self._get_states(
            alchem_system, positions, settings, lambdas, solv_comp
        )

        # 8. Create the multistate reporter & create PDB
        reporter = self._get_reporter(
            omm_topology,
            positions,
            settings["simulation_settings"],
            settings["output_settings"],
        )

        # TODO: delete all this once this changes upstream soon
        # Wrap in try/finally to avoid memory leak issues
        try:
            # 12. Get context caches
            energy_ctx_cache, sampler_ctx_cache = self._get_ctx_caches(
                settings["engine_settings"]
            )

            # 13. Get integrator
            integrator = self._get_integrator(
                settings["integrator_settings"],
                settings["simulation_settings"],
            )

            # 14. Get sampler
            sampler = self._get_sampler(
                integrator,
                reporter,
                settings["simulation_settings"],
                settings["thermo_settings"],
                cmp_states,
                sampler_states,
                energy_ctx_cache,
                sampler_ctx_cache,
            )

            # 15. Run simulation
            unit_result_dict = self._run_simulation(sampler, reporter, settings, dry)

        finally:
            # close reporter when you're done to prevent file handle clashes
            reporter.close()

            # clear GPU context
            # Note: use cache.empty() when openmmtools #690 is resolved
            for context in list(energy_ctx_cache._lru._data.keys()):
                del energy_ctx_cache._lru._data[context]
            for context in list(sampler_ctx_cache._lru._data.keys()):
                del sampler_ctx_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(
                openmmtools.cache.global_context_cache._lru._data.keys()
            ):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_ctx_cache, energy_ctx_cache

            # Keep these around in a dry run so we can inspect things
            if not dry:
                del integrator, sampler

        if not dry:
            nc = self.shared_basepath / settings["output_settings"].output_filename
            chk = settings["output_settings"].checkpoint_storage_filename
            return {
                "nc": nc,
                "last_checkpoint": chk,
                **unit_result_dict,
            }
        else:
            return {"debug": {"sampler": sampler}}

    def _execute(
        self,
        ctx: gufe.Context,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "simtype": self._simtype,
            **outputs,
        }
