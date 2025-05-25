# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
import openfe
from openmmtools.tests.test_alchemy import (
    compare_system_energies,
    check_noninteracting_energy_components,
    check_interacting_energy_components,
    overlap_check,
)
from openmmtools.alchemy import AlchemicalRegion
from openff.toolkit import ForceField, Molecule
from openff.interchange.interop.openmm import to_openmm_positions
from pontibus.components.extended_solvent_component import ExtendedSolventComponent
from pontibus.utils.molecules import WATER
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.utils.system_creation import (
    interchange_packmol_creation
)
from pontibus.utils.experimental_absolute_factory import (
    ExperimentalAbsoluteAlchemicalFactory,
)


@pytest.fixture(scope="module")
def vinyl_chloride():
    m = Molecule.from_smiles("C=CCl")
    m.generate_conformers(n_conformers=1)
    m.assign_partial_charges(partial_charge_method="gasteiger")
    return m


@pytest.fixture(scope="module")
def water_off():
    return WATER.to_openff()


class TestVSiteEnergies:
    @pytest.fixture(scope='class')
    def interchange(self, vinyl_chloride, vsite_offxml, water_off):
        solute = openfe.SmallMoleculeComponent.from_openff(vinyl_chloride)
        interchange, comp_resids = interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(
                forcefields=[vsite_offxml],
            ),
            solvation_settings=PackmolSolvationSettings(
                number_of_solvent_molecules=1000,
                solvent_padding=None,
            ),
            smc_components={solute: vinyl_chloride},
            protein_component=None,
            solvent_component=ExtendedSolventComponent(),
            solvent_offmol=water_off,
        )

        return interchange

    @pytest.fixture(scope='class')
    def omm_system(self, interchange):
        return interchange.to_openmm_system()

    @pytest.fixture(scope='class')
    def positions(self, interchange):
        return to_openmm_positions(interchange, include_virtual_sites=True)

    @pytest.fixture(scope='class')
    def alchemical_region(self):
        alchemical_indices = [0, 1, 2, 3, 4, 5, 3006]
        return AlchemicalRegion(alchemical_atoms=alchemical_indices)

    @pytest.fixture(scope='class')
    def alchemical_system(self, omm_system, alchemical_region):
        alchemical_factory = ExperimentalAbsoluteAlchemicalFactory()
        return alchemical_factory.create_alchemical_system(
            omm_system, alchemical_region
        )

    def test_compare_energies(self, omm_system, alchemical_system, alchemical_region, positions):
        compare_system_energies(
            omm_system,
            alchemical_system,
            alchemical_region,
            positions
        )

    def test_noninteracting_energies(self, omm_system, alchemical_system, alchemical_region, positions):
        check_noninteracting_energy_components(
            omm_system,
            alchemical_system,
            alchemical_region,
            positions,
        )

    def test_interacting_energies(self, omm_system, alchemical_system, alchemical_region, positions):
        check_interacting_energy_components(
            omm_system,
            alchemical_system,
            alchemical_region,
            positions,
        )

    @pytest.mark.slow
    def test_overlap(self, omm_system, alchemical_system, positions):
        overlap_check(
            omm_system,
            alchemical_system,
            positions,
            cached_trajectory_filename=None,
            name="test"
        )
