# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import logging
from typing import Any

import gufe
import numpy.typing as npt
import openmm
from gufe import (
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.settings import SettingsBaseModel
from openfe.protocols.openmm_afe.base import BaseAbsoluteUnit
from openfe.protocols.openmm_utils import charge_generation
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    OpenFFPartialChargeSettings,
)
from openfe.utils import log_system_probe, without_oechem_backend
from openff.interchange.interop.openmm import to_openmm_positions
from openff.toolkit import Molecule as OFFMolecule
from openmm import app
from openmmtools.alchemy import (
    AbsoluteAlchemicalFactory,
    AlchemicalRegion,
)

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation.settings import PackmolSolvationSettings
from pontibus.utils.experimental_absolute_factory import (
    ExperimentalAbsoluteAlchemicalFactory,
)
from pontibus.utils.system_creation import interchange_packmol_creation

logger = logging.getLogger(__name__)


class BaseASFEUnit(BaseAbsoluteUnit):
    _simtype: str

    @staticmethod
    def _get_and_charge_solvent_offmol(
        solvent_component: SolventComponent | ExtendedSolventComponent,
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

        Notes
        -----
        * If created from a smiles, the solvent will be assigned
          a single conformer through `Molecule.generate_conformers`.
        """
        # Get the solvent offmol
        if isinstance(solvent_component, ExtendedSolventComponent):
            solvent_offmol = solvent_component.solvent_molecule.to_openff()  # type: ignore[union-attr]
        else:
            # If not, we create the solvent from smiles
            # We generate a single conformer to avoid packing issues
            solvent_offmol = OFFMolecule.from_smiles(solvent_component.smiles)
            solvent_offmol.generate_conformers(n_conformers=1)

        # In-place assign solvent offmol charges if necessary
        # Note: we don't enforce partial charge assignment to avoid
        # cases where we want to rely on library charges instead.
        if solvation_settings.assign_solvent_charges:
            charge_generation.assign_offmol_partial_charges(
                offmol=solvent_offmol,
                overwrite=False,
                method=partial_charge_settings.partial_charge_method,
                toolkit_backend=partial_charge_settings.off_toolkit_backend,
                generate_n_conformers=partial_charge_settings.number_of_conformers,
                nagl_model=partial_charge_settings.nagl_model,
            )

        return solvent_offmol

    @staticmethod
    def _validate_vsites(system: openmm.System, integrator_settings: IntegratorSettings) -> None:
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
                    "Simulations with virtual sites without velocity reassignments are unstable"
                )
                raise ValueError(errmsg)

    def _get_omm_objects(
        self,
        settings: dict[str, SettingsBaseModel],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
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
        omm_topology = interchange.to_openmm_topology(collate=False)
        omm_system = interchange.to_openmm_system(
            hydrogen_mass=settings["forcefield_settings"].hydrogen_mass
        )

        # Pull out the CMMotionRemover
        # TODO: add test that checks the number of forces
        for idx in reversed(range(omm_system.getNumForces())):
            force = omm_system.getForce(idx)
            if isinstance(force, openmm.CMMotionRemover):
                omm_system.removeForce(idx)

        positions = to_openmm_positions(interchange, include_virtual_sites=True)

        # Post creation system validation
        self._validate_vsites(omm_system, settings["integrator_settings"])

        return omm_topology, omm_system, positions, comp_resids

    def _get_experimental_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]],
    ) -> tuple[ExperimentalAbsoluteAlchemicalFactory, openmm.System, list[int]]:
        """
        Get an alchemically modified system and its associated factory using
        the ExperimentalAbsoluteAlchemicalFactory,

        Parameters
        ----------
        topology : openmm.Topology
          Topology of OpenMM System.
        system : openmm.System
          System to alchemically modify.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : dict[str, list[Component]]
          A dictionary of alchemical components for each end state.

        Returns
        -------
        alchemical_factory : AbsoluteAlchemicalFactory
          Factory for creating an alchemically modified system.
        alchemical_system : openmm.System
          Alchemically modified system
        alchemical_indices : list[int]
          A list of atom indices for the alchemically modified
          species in the system.

        Notes
        -----
        In theory, this option should allow for virtual sites.

        TODO
        ----
        * Add support for all alchemical factory options
        """
        alchemical_indices = self._get_alchemical_indices(topology, comp_resids, alchem_comps)

        alchemical_region = AlchemicalRegion(
            alchemical_atoms=alchemical_indices,
        )

        alchemical_factory = ExperimentalAbsoluteAlchemicalFactory()
        alchemical_system = alchemical_factory.create_alchemical_system(system, alchemical_region)

        return alchemical_factory, alchemical_system, alchemical_indices

    def _get_alchemical_system(
        self,
        topology: app.Topology,
        system: openmm.System,
        comp_resids: dict[Component, npt.NDArray],
        alchem_comps: dict[str, list[Component]],
    ) -> tuple[AbsoluteAlchemicalFactory, openmm.System, list[int]]:
        """
        Get an alchemically modified system and its associated factory.

        If the experimental settings are turned on, will return an
        ExperimentalAlchemicalFactory via
        :meth:`_get_experimental_alchemical_system`.

        Parameters
        ----------
        topology : openmm.Topology
          Topology of OpenMM System.
        system : openmm.System
          System to alchemically modify.
        comp_resids : dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : dict[str, list[Component]]
          A dictionary of alchemical components for each end state.

        Returns
        -------
        alchemical_factory : AbsoluteAlchemicalFactory
          Factory for creating an alchemically modified system.
        alchemical_system : openmm.System
          Alchemically modified system
        alchemical_indices : list[int]
          A list of atom indices for the alchemically modified
          species in the system.
        """
        if self._inputs["protocol"].settings.alchemical_settings.experimental:
            return self._get_experimental_alchemical_system(
                topology,
                system,
                comp_resids,
                alchem_comps,
            )
        else:
            return super()._get_alchemical_system(
                topology,
                system,
                comp_resids,
                alchem_comps,
            )

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
