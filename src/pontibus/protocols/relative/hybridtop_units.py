# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
ProtocolUnit implementations for the HybridTopProtocol.
"""

import logging
import warnings

import numpy as np
import numpy.typing as npt
import openmm
from gufe import (
    ChemicalSystem,
    LigandAtomMapping,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.settings import (
    SettingsBaseModel,
    ThermoSettings,
)
from gufe.settings.typing import GufeArrayQuantity
from openfe.protocols.openmm_rfe import _rfe_utils
from openfe.protocols.openmm_rfe.hybridtop_units import HybridTopologySetupUnit
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
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

from pontibus.protocols.solvation.base import _get_and_charge_solvent_offmol
from pontibus.utils.settings import (
    InterchangeFFSettings,
)
from pontibus.utils.system_creation import (
    _get_comp_resids,
    interchange_system_creation,
)
from pontibus.utils.system_manipulation import (
    adjust_system,
    copy_interchange_with_replacement,
)

logger = logging.getLogger(__name__)


class HybridTopProtocolSetupUnit(HybridTopologySetupUnit):
    """
    Setup unit for the HybridTopProtocol.

    Overrides :meth:`_get_omm_objects` to use Interchange-based
    parameterization instead of OpenMM's SystemGenerator.
    """

    @staticmethod
    def _check_position_overlap(
        mapping: dict[str, dict[int, int]],
        positionsA: GufeArrayQuantity,
        positionsB: GufeArrayQuantity,
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
        threshold : openff.units.Quantity
          The maximum deviation allowed before an error or warning is raised.

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
    def _get_interchanges(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
        protein_component,
        solvent_component: SolventComponent | None,
        forcefield_settings: InterchangeFFSettings,
        solvation_settings,
        charge_settings,
    ) -> tuple[Interchange, Interchange, dict[str, npt.NDArray]]:
        """
        Build Interchange objects for state A and state B.

        Parameters
        ----------
        stateA : ChemicalSystem
          ChemicalSystem defining end state A.
        stateB : ChemicalSystem
          ChemicalSystem defining end state B.
        small_mols : dict[SmallMoleculeComponent, OFFMolecule]
          All small molecule components for both states (flat dict).
        protein_component : ProteinComponent | None
          The protein component, if present.
        solvent_component : SolventComponent | None
          The solvent component, if present.
        forcefield_settings : InterchangeFFSettings
          Force field settings.
        solvation_settings : PackmolSolvationSettings or similar
          Solvation settings.
        charge_settings : OpenFFPartialChargeSettings
          Partial charge settings.

        Returns
        -------
        interA : Interchange
          Interchange for state A.
        interB : Interchange
          Interchange for state B (built by replacing ligand A with ligand B
          in interA's solvated box).
        alchem_resids : dict[str, npt.NDArray]
          Residue indices for the alchemical ligand in each state.
        """

        def _filter_small_mols(smols, state):
            return {smc: offmol for smc, offmol in smols.items() if state.contains(smc)}

        stateA_small_mols = _filter_small_mols(small_mols, stateA)
        stateB_small_mols = _filter_small_mols(small_mols, stateB)

        # Get the alchemical components.
        # We assume a single alchemical component, since we check for this
        # early in the Protocol.
        state_diff = stateA.component_diff(stateB)
        alchem_A = state_diff[0][0]
        alchem_B = state_diff[1][0]

        # Get solvent offmol if necessary
        if solvent_component is not None:
            solvent_offmol = _get_and_charge_solvent_offmol(
                solvent_component,
                solvation_settings,
                charge_settings,
            )
        else:
            solvent_offmol = None

        # Build state A interchange
        with without_oechem_backend():
            interA, comp_residsA = interchange_system_creation(
                ffsettings=forcefield_settings,
                solvation_settings=solvation_settings,
                smc_components=stateA_small_mols,
                protein_component=protein_component,
                solvent_component=solvent_component,
                solvent_offmol=solvent_offmol,
            )

        stateB_charged_mols = list(stateB_small_mols.values())
        if solvent_component is not None and solvation_settings.assign_solvent_charges:
            stateB_charged_mols.append(solvent_offmol)

        # Tag molB so copy_interchange_with_replacement can identify it
        molB = stateB_small_mols[alchem_B]
        molB.properties["key"] = str(alchem_B.key)

        # Build state B interchange by swapping ligand A → ligand B in the
        # pre-solvated state A box
        with without_oechem_backend():
            interB = copy_interchange_with_replacement(
                interchange=interA,
                del_mol=stateA_small_mols[alchem_A],
                insert_mol=molB,
                ffsettings=forcefield_settings,
                charged_molecules=stateB_charged_mols,
                protein_component=protein_component,
            )

        comp_residsB = _get_comp_resids(
            interchange=interB,
            smc_components=stateB_small_mols,
            solvent_component=solvent_component,
            protein_component=protein_component,
        )

        alchem_resids = {
            "stateA": comp_residsA[alchem_A],
            "stateB": comp_residsB[alchem_B],
        }

        return interA, interB, alchem_resids

    @staticmethod
    def _omm_from_interchange(
        interchange: Interchange,
        forcefield_settings: InterchangeFFSettings,
        thermo_settings: ThermoSettings,
        integrator_settings: IntegratorSettings,
        solvent_component: SolventComponent | None,
    ) -> tuple[Topology, omm_unit.Quantity, System]:
        """
        Extract OpenMM Topology, positions, and System from an Interchange.

        Parameters
        ----------
        interchange : Interchange
          The Interchange object to convert.
        forcefield_settings : InterchangeFFSettings
          Force field settings (for hydrogen mass).
        thermo_settings : ThermoSettings
          Thermodynamic settings (for barostat temperature/pressure).
        integrator_settings : IntegratorSettings
          Integrator settings (for barostat frequency).
        solvent_component : SolventComponent | None
          The solvent component; if not None a barostat is added.

        Returns
        -------
        topology : openmm.app.Topology
          OpenMM Topology for the system.
        positions : openmm.unit.Quantity
          Positions of all particles in the system.
        system : openmm.System
          OpenMM System with hydrogen mass repartitioning applied, and a
          MonteCarloBarostat if ``solvent_component`` is not ``None``.
        """

        def _get_barostat(
            solvent_component: SolventComponent | None,
            thermo_settings: ThermoSettings,
            integrator_settings: IntegratorSettings,
        ) -> MonteCarloBarostat | None:
            """
            Helper function to add a MonteCarloBarostat
            if a solvent component is present.
            """
            if solvent_component is None:
                return None

            return MonteCarloBarostat(
                to_openmm(thermo_settings.pressure),
                to_openmm(thermo_settings.temperature),
                integrator_settings.barostat_frequency.m,
            )

        topology = interchange.to_openmm_topology(collate=True)
        positions = to_openmm_positions(interchange, include_virtual_sites=True)
        system = interchange.to_openmm_system(hydrogen_mass=forcefield_settings.hydrogen_mass)

        adjust_system(
            system=system,
            remove_force_types=CMMotionRemover,
            add_forces=_get_barostat(
                solvent_component=solvent_component,
                thermo_settings=thermo_settings,
                integrator_settings=integrator_settings,
            ),
        )
        return topology, positions, system

    def _get_omm_objects(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: LigandAtomMapping,
        settings: dict[str, SettingsBaseModel],
        protein_component: ProteinComponent | None,
        solvent_component: SolventComponent | None,
        small_mols: dict[SmallMoleculeComponent, OFFMolecule],
    ) -> tuple[
        openmm.System,
        openmm.app.Topology,
        openmm.unit.Quantity,
        openmm.System,
        openmm.app.Topology,
        openmm.unit.Quantity,
        dict[str, dict[int, int]],
    ]:
        """
        Get OpenMM objects for both end states using Interchange.

        Overrides :meth:`HybridTopologySetupUnit._get_omm_objects` to use
        OpenFF Interchange for parameterization instead of OpenMM's
        SystemGenerator.

        Parameters
        ----------
        stateA : ChemicalSystem
          ChemicalSystem defining end state A.
        stateB : ChemicalSystem
          ChemicalSystem defining end state B.
        mapping : LigandAtomMapping
          The mapping for alchemical components between state A and B.
        settings : dict[str, SettingsBaseModel]
          Settings for the transformation.
        protein_component : ProteinComponent | None
          The common ProteinComponent between the end states, if there is one.
        solvent_component : SolventComponent | None
          The common SolventComponent between the end states, if there is one.
        small_mols : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
          The small molecules for both end states.

        Returns
        -------
        stateA_system : openmm.System
          OpenMM System for state A.
        stateA_topology : openmm.app.Topology
          OpenMM Topology for the state A System.
        stateA_positions : openmm.unit.Quantity
          Positions of all particles in state A System.
        stateB_system : openmm.System
          OpenMM System for state B.
        stateB_topology : openmm.app.Topology
          OpenMM Topology for the state B System.
        stateB_positions : openmm.unit.Quantity
          Positions of all particles in state B System.
        system_mappings : dict[str, dict[int, int]]
          Dictionary of mappings defining the correspondence between
          the two state Systems.
        """
        if self.verbose:
            self.logger.info("Parameterizing systems.")

        interA, interB, alchem_resids = self._get_interchanges(
            stateA=stateA,
            stateB=stateB,
            small_mols=small_mols,
            protein_component=protein_component,
            solvent_component=solvent_component,
            forcefield_settings=settings["forcefield_settings"],
            solvation_settings=settings["solvation_settings"],
            charge_settings=settings["charge_settings"],
        )

        stateA_topology, stateA_positions, stateA_system = self._omm_from_interchange(
            interchange=interA,
            forcefield_settings=settings["forcefield_settings"],
            thermo_settings=settings["thermo_settings"],
            integrator_settings=settings["integrator_settings"],
            solvent_component=solvent_component,
        )
        stateB_topology, stateB_positions, stateB_system = self._omm_from_interchange(
            interchange=interB,
            forcefield_settings=settings["forcefield_settings"],
            thermo_settings=settings["thermo_settings"],
            integrator_settings=settings["integrator_settings"],
            solvent_component=solvent_component,
        )

        system_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system,
            stateA_topology,
            alchem_resids["stateA"],
            stateB_system,
            stateB_topology,
            alchem_resids["stateB"],
            # non-optional setting for this method
            fix_constraints=True,
        )

        self._check_position_overlap(
            system_mappings,
            from_openmm(stateA_positions),
            from_openmm(stateB_positions),
        )

        if settings["alchemical_settings"].explicit_charge_correction:
            # Note: we don't need to modify the positions after this
            # since stateB retains the hydrogens but as dummy atoms
            charge_difference = mapping.get_alchemical_charge_difference()
            alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
                stateA_topology,
                from_openmm(stateA_positions).m,
                charge_difference,
                settings["alchemical_settings"].explicit_charge_correction_cutoff,
            )
            _rfe_utils.topologyhelpers.handle_alchemical_waters(
                alchem_water_resids,
                stateB_topology,
                stateB_system,
                system_mappings,
                charge_difference,
                solvent_component,
            )

        return (
            stateA_system,
            stateA_topology,
            stateA_positions,
            stateB_system,
            stateB_topology,
            stateB_positions,
            system_mappings,
        )

    def _subsample_topology(
        self,
        hybrid_topology,
        hybrid_positions,
        output_selection: str,
        output_filename: str,
        atom_classes: dict,
    ):
        """
        Override of parent to additionally write the full hybrid topology
        as ``full_hybrid_system.pdb``.
        """
        selection_indices = super()._subsample_topology(
            hybrid_topology=hybrid_topology,
            hybrid_positions=hybrid_positions,
            output_selection=output_selection,
            output_filename=output_filename,
            atom_classes=atom_classes,
        )

        # Write the full (unsubsampled) hybrid topology
        super()._subsample_topology(
            hybrid_topology=hybrid_topology,
            hybrid_positions=hybrid_positions,
            output_selection="all",
            output_filename=f"full_{output_filename}",
            atom_classes=atom_classes,
        )

        return selection_indices
