from openfe.protocols.openmm_afe.base import BaseAbsoluteUnit
from openfe.utils import without_oechem_backend
from openfe.protocols.openmm_utils import (
    charge_generation,
)
from openff.toolkit import Molecule as OFFMolecule
from openff.interchange.components._packmol import (
    solvate_topology,
    solvate_topology_nonwater,
    RHOMBIC_DODECAHEDRON,
    UNIT_CUBE,
)


def _set_offmol_resname(
    offmol: OFFMolecule,
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
        a.metadata["residue_name"] = resname


def _get_offmol_resname(offmol: OFFMolecule) -> Optional[str]:
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
                resname = a.metadata["residue_name"]
            except KeyError:
                return None

        if resname != a.metadata["residue_name"]:
            wmsg = f"Inconsistent residue name in OFFMol: {offmol} "
            logger.warning(wmsg)
            return None

    return resname


def interchange_packmol_creation(
    ffsettings: InterchangeFFSettings,
    solvation_settings: PackmolSolvationSettings,
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
    protein_component: Optional[ProteinComponent],
    solvent_component: Optional[SolventComponent],
    solvent_offmol: Optional[OFFMolecule],
) -> tuple[Interchange, dict[str, npt.NDArray]]:
    """
    Create an Interchange object for a given system, solvating with
    packmol where necessary.

    Parameters
    ----------
    ffsettings : InterchangeFFSettings
      Settings defining how the force field is applied.
    solvation_settings : PackmolSolvationSettings
      Settings defining how the system will be solvated.
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]`
      Solute SmallMoleculeComponents.
    protein_component : Optional[ProteinComponent]
      Protein component of the system, if any.
    solvent_component : Optional[SolventComponent]
      Solvent component of the system, if any.
    solvent_offmol : Optional[openff.toolkit.Molecule]
      OpenFF Molecule defining the solvent, if necessary

    Returns
    -------
    Interchange : openff.interchange.Interchange
      Interchange object for the created system.
    comp_resids : dict[str, npt.NDArray]
      A dictionary defining the residue indices matching
      various components in the system.
    """

    # 1. Component validations
    # Adding protein components is not currently supported
    if protein_component is not None:
        errmsg = (
            "Creation of systems solely with Interchange "
            "using ProteinComponents is not currently supported"
        )
        raise ValueError(errmsg)

    # TODO: work out ways to deal with the addition of counterions
    if solvent_component is not None:
        if (
            solvent_component.neutralize
            or solvent_component.ion_concentration > 0 * offunit.molar
        ):
            errmsg = (
                "Adding counterions using packmol solvation "
                "is currently not supported"
            )
            raise ValueError(errmsg)

        if solvent_offmol is None:
            errmsg = "A solvent offmol must be passed to solvate a " "system!"
            raise ValueError(errmsg)

    # 2. Get the force field object
    # force_fields is a list so we unpack it
    force_field = ForceField(*ffsettings.forcefields)

    # We also set nonbonded cutoffs whilst we are here
    # TODO: check what this means for nocutoff simulations
    force_field["vdW"].cutoff = ffsettings.nonbonded_cutoff
    force_field["Electrostatics"].cutoff = ffsettings.nonbonded_cutoff

    # 3. Asisgn residue names so we can track our components in the generated
    # topology.

    # Note: comp_resnames is dict[str, tuple[Component, list]] where the final list
    # is to append residues later on
    # TODO: we should be able to rely on offmol equality in the same way that
    # intecharge does
    comp_resnames: dict[str, tuple[Component, list[Any]]] = {}

    # If we have solvent, we set its residue name
    if solvent_component is not None:
        offmol_resname = _get_offmol_resname(solvent_offmol)
        if offmol_resname is None:
            offmol_resname = "SOL"
            _set_offmol_resname(solvent_offmol, offmol_resname)
        comp_resnames[offmol_resname] = (solvent_component, [])

    # A store of residue names to replace residue names if they aren't unique
    resnames_store = ["".join(i) for i in product(ascii_uppercase, repeat=3)]

    for comp, offmol in smc_components.items():
        off_resname = _get_offmol_resname(offmol)
        if off_resname is None or off_resname in comp_resnames:
            # warn that we are overriding clashing molecule resnames
            if off_resname in comp_resnames:
                wmsg = (
                    f"Duplicate residue name {off_resname}, "
                    "duplicate will be renamed"
                )
                logger.warning(wmsg)

            # just loop through and pick up a name that doesn't exist
            while (off_resname in comp_resnames) or (off_resname is None):
                off_resname = resnames_store.pop(0)

        wmsg = f"Setting component {comp} residue name to {off_resname}"
        logger.warning(wmsg)
        _set_offmol_resname(offmol, off_resname)
        comp_resnames[off_resname] = [comp, []]

    # 4. Create an OFF Topology from the smcs
    # Note: this is the base no solvent case!
    topology = Topology.from_molecules(*smc_components.values())

    # Also create a list of charged molecules for later use
    charged_mols = [*smc_components.values()]

    # 5. Solvent case
    if solvent_component is not None:

        # Append to charged molcule to charged_mols if we want to
        # otherwise we rely on library charges
        if solvation_settings.assign_solvent_charges:
            charged_mols.append(solvent_offmol)

        # Pick up the user selected box shape
        box_shape = {
            "cube": UNIT_CUBE,
            "dodecahedron": RHOMBIC_DODECAHEDRON,
        }[solvation_settings.box_shape.lower()]

        # TODO: switch back to normal pack_box and allow n_solvent
        # Create the topology
        topology = solvate_topology_nonwater(
            topology=topology,
            solvent=solvent_offmol,
            padding=solvation_settings.solvent_padding,
            box_shape=box_shape,
        )

    # Assign residue indices to each entry in the OFF topology
    for molecule_index, molecule in enumerate(topology.molecules):
        for atom in molecule.atoms:
            atom.metadata["residue_number"] = molecule_index

        # Get the residue name and store the index in comp resnames
        resname = _get_offmol_resname(molecule)
        comp_resnames[resname][1].append(molecule_index)

    # Now create the component_resids dictionary properly
    comp_resids = {}
    for entry in comp_resnames.values():
        comp = entry[0]
        comp_resids[comp] = np.array(entry[1])

    interchange = force_field.create_interchange(
        topology=topology,
        charge_from_molecules=charged_mols,
    )

    return interchange, comp_resids


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
        has_virtual_sites = False
        for ix in range(system.getNumParticles()):
            if system.isVirtualSite(ix):
                has_virtual_sites = True

        if hybrid_factory.has_virtual_sites:
            if not integrator_settings.reassign_velocities:
                errmsg = (
                    "Simulations with virtual sites without velocity "
                    "reassignments are unstable in openmmtools"
                )
                raise ValueError(errmsg)

    def _get_omm_objects(
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
            solvent_offmol = _get_and_charge_solvent_offmol(
                solvent_component,
                settings["solvation_settings"],
                settings["charge_settings"],
            )
        else:
            solvent_offmol = None

        # Create your interchange object
        with without_oechem_backend():
            interchange, comp_resids = interchange_packmol_creation(
                settings["forcefield_settings"],
                settings["solvation_settings"],
                protein_component,
                solvent_component,
                solvent_offmol,
                smc_components,
            )

        # Get omm objects back
        omm_topology = interchange.to_openmm_topology()
        omm_system = interchange.to_openmm_system(
            hydrogen_mass=settings["forcefield_settings"].hydrogen_mass
        )
        positions = interchange.positions.to_openmm()

        # Post creation system validation
        _validate_vsites(omm_system, settings["integrator_settings"])

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
