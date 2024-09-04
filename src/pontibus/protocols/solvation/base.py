from openfe.protocols.openmm_afe.base import BaseAbsoluteUnit
from openfe.utils import without_oechem_backend


def _set_offmol_resname(
    offmol: OFFMol,
    resname: str,
) -> None:
    """
    Helper method to set offmol residue names

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to assign a residue name to.
    resname : str
      Residue name to be set.

    Returns
    -------
    None
    """
    for a in offmol.atoms:
        a.metadata['residue_name'] = resname


def _get_offmol_resname(offmol: OFFMol) -> Optional[str]:
    """
    Helper method to get an offmol's residue name and make sure it is
    consistent across all atoms in the Molecule.

    Parameters
    ----------
    offmol : openff.toolkit.Molecule
      Molecule to get the residue name from.

    Returns
    -------
    resname : Optional[str]
      Residue name of the molecule. ``None`` if the Molecule
      does not have a residue name, or if the residue name is
      inconsistent across all the atoms.
    """
    resname: Optional[str] = None
    for a in offmol.atoms:
        if resname is None:
            try:
                resname = a.metadata['residue_name']
            except KeyError:
                return None

        if resname != a.metadata['residue_name']:
            wmsg = (f"Inconsistent residue name in OFFMol: {offmol} ")
            logger.warning(wmsg)
            return None

    return resname


def interchange_system_creation(
    ffsettings: InterchangeFFSettings,
    solvation_settings: PackmolSolvationSettings,
    protein_component: Optional[ProteinComponent],
    solvent_component: Optional[SolventComponent],
    solvent_smcs: Optional[dict[SmallMoleculeComponent, OFFMolecule]],
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
) -> tuple[app.Topology, openmm.System, openmm.unit.Quantity, dict[str, npt.NDArray]]:

    # 1. Component validations
    if protein_component is not None:
        errmsg = (
            "Creation of systems solely with Interchange "
            "using ProteinComponents is not currently supported"
        )
        raise ValueError(errmsg)

    # 2. Get the force field object
    # force_fields is a list so we unpack it
    force_field = ForceField(*ffsettings.force_fields)

    # We also set nonbonded cutoffs whilst we are here
    # TODO: check what this means for nocutoff simulations
    force_field['vdW'].cutoff = ffsettings.nonbonded_cutoff
    force_field['Electrostatics'].cutoff = ffsettings.nonbonded_cutoff

    # 3. Asisgn residue names so we can track our components in the generated
    # topology.

    # Note: comp_resnames is dict[str, tuple[Component, list]] where the final list
    # is to append residues later on
    # TODO: can we rely on offmol equality here instead?
    comp_resnames: dict[str, tuple[Component, list[Any]]] = {}
    if solvent_component is not None:
        comp_resnames['SOL'] = (solvent_component, [])

    # A store of residue names to replace residue names if they aren't unique
    resnames_store = [''.join(i) for i in product(ascii_uppercase, repeat=3)]

    for comp, offmol in smc_components.items():
        off_resname = _get_offmol_resname(offmol)
        if off_resname is None or off_resname in comp_resnames:
            # warn that we are overriding clashing molecule resnames
            if off_resname in comp_resnames:
                wmsg = (f"Duplicate residue name {off_resname}, "
                        "duplicate will be renamed")
                logger.warning(wmsg)

            # just loop through and pick up a name that doesn't exist
            while (off_resname in comp_resnames) or (off_resname is None):
                off_resname = resnames_store.pop(0)

        wmsg = f"Setting component {comp} residue name to {off_resname}"
        logger.warning(wmsg)
        _set_offmol_resname(offmol, off_resname)
        comp_resnames[off_resname] = [comp, []]

    # 4. No solvent case
    # If we don't have any solvent, just go ahead and create a topology from
    # the SMCS
    if solvent_component is None:
        topology = Topology.from_molecules(*smc_components.values())
    else:
        # TODO: add back solvate_topology code once fixed
        if (solvent_component.neutralize
            or solvent_component.ion_concentration > 0 * unit.molar):
            ... pack_box things


        if solvent_component.smiles == 'O'
        from openff.interchange.components._packmol import solvate_topology, UNIT_CUBE

            # might want to split out depending on if pack_box or solvate_topology is used
            # TODO: Actually use the right settings here
            # TODO: Either update solvate_topology to actually use non-water solvent
            #       or use pack_box to add the solvent or make a new function altogether

            # lost solvent settings include
            #  * positive_ion
            #  * negative_ion
            #  * neutralize
            # lost solvation settings include
            #  * solvent_model
            assert solvent_component.positive_ion == "Na+"
            assert solvent_component.negative_ion == "Cl-"

            assert len(smc_components) == 1

            topology = solvate_topology(
                topology=[*smc_components.values()][0].to_topology(),
                nacl_conc = solvent_component.ion_concentration,
                padding = settings['solvation_settings'].solvent_padding,
                box_shape=UNIT_CUBE,
            )

            water = OFFMolecule.from_smiles("O")
            na = OFFMolecule.from_smiles("[Na+]")
            cl = OFFMolecule.from_smiles("[Cl-]")

            for molecule_index, molecule in enumerate(topology.molecules):
                for atom in molecule.atoms:
                    atom.metadata['residue_number'] = molecule_index

                if molecule.n_atoms == [*smc_components.values()][0].n_atoms:
                    # this is probably UNK, but just leave it be I guess
                    continue
                # molecules don't know their residue metadata, so need to set on each atom
                # https://github.com/openforcefield/openff-toolkit/issues/1554
                elif molecule.is_isomorphic_with(water):
                    for atom in molecule.atoms:
                        atom.metadata['residue_name'] = "WAT"
                elif molecule.is_isomorphic_with(na):
                    for atom in molecule.atoms:
                        atom.metadata['residue_name'] = "Na"
                elif molecule.is_isomorphic_with(cl):
                    for atom in molecule.atoms:
                        atom.metadata['residue_name'] = "Cl"
                else:
                    raise Exception("Found unexpected molecule in solvated topology")

            component_resids[solvent_component] = np.arange(1, topology.n_molecules)


    # TODO: doesn't work with solvent smcs
    interchange = force_field.create_interchange(
        topology=topology,
        charge_from_molecules=[*smc_components.values()],
    )

    return (
        interchange.to_openmm_topology(),
        interchange.to_openmm_system(
            hydrogen_mass=ffsettings.hydrogen_mass,
        ),
        interchange.positions.to_openmm(),
        component_resids,
    )


class BaseSFEUnit(BaseAbsoluteUnit):
    def _get_omm_objects(
        settings: dict[str, SettingsBaseModel],
        protein_component: Optional[ProteinComponent],
        solvent_component: Optional[SolventComponent],
        solvent_smcs: Optional[dict[SmallMoleculeComponent, OFFMolecule]],
        smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> tuple[app.Topology, openmm.System, openmm.unit.Quantity, dict[str, npt.NDArray]]:
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
        solvent_smcs : Optional[dict[SmallMoleculeComponent, OFFMolecule]]
          SmallMoleculeComponents associated with the solvent component
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
        self._assign_partial_charges(settings['charge_settings'], smc_components)
        
        if solvent_smcs is not None:
            self._assign_partial_charges(settings['charge_settings'], solvent_smcs)

        with without_oechem_backend():
            return interchange_system_creation(
                settings['forcefield_settings'],
                protein_component,
                solvent_component,
                solvent_smcs,
                smc_components,
            )

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

        # 1. Get settings
        settings = self._handle_settings()

        # 2. Get components
        ## solv_smcs is a poor hack, to be refined
        alchem_comps, solv_comp, solv_smcs, prot_comp, smc_comps = self._get_components(
            settings
        )

        # 3. Get OpenMM topology, positions and system
        omm_topology, omm_system, positions, comp_resids = self._get_omm_objects(
            settings, prot_comp, solv_comp, solv_smcs, smc_comps,
        )

        # 4. Pre-equilbrate System (Test + Avoid NaNs + get stable system)
        positions = self._pre_equilibrate(
            omm_system, omm_topology, positions, settings, dry
        )

        # 5. Get lambdas
        lambdas = self._get_lambda_schedule(settings)

        # 6. Get alchemical system
        if settings['alchemical_settings'].experimental:
            raise ValueError("experimental factory code is not yet implemented")
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
