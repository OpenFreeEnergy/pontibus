from openfe.protocols.openmm_afe.base import BaseAbsoluteUnit
from openfe.utils import without_oechem_backend


def interchange_system_creation(...):
    if protein_component is not None:
        errmsg = (
            "Creation of systems solely with Interchange "
            "using ProteinComponents is not currently supported"
        )
        raise ValueError(errmsg)




class BaseAFEUnit(BaseAbsoluteUnit):
    def _get_omm_objects(

    ):
        if self.verbose:
            self.logger.info("Parameterizing system")

        # Set partial charges for all smcs
        self._assign_partial_charges(settings['charge_settings'], smc_components)

        with without_oechem_backend():
            interchange_system_creation()

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
            system_modeller, system_generator, list(smc_comps.values())
        )

        # 4. Pre-equilbrate System (Test + Avoid NaNs + get stable system)
        positions = self._pre_equilibrate(
            omm_system, omm_topology, positions, settings, dry
        )

        # 5. Get lambdas
        lambdas = self._get_lambda_schedule(settings)

        # 6. Get alchemical system
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
