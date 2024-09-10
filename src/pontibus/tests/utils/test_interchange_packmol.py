# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
import logging
from openff.toolkit import Molecule
from openff.units import unit
from gufe import SmallMoleculeComponent, SolventComponent
from pontibus.protocols.solvation.settings import (
    InterchangeFFSettings,
    PackmolSolvationSettings,
)
from pontibus.components.extended_solvent_component import (
    ExtendedSolventComponent,
)
from pontibus.utils.system_creation import (
    interchange_packmol_creation,
    _set_offmol_resname,
    _get_offmol_resname,
)


LOGGER = logging.getLogger(__name__)


@pytest.fixture()
def smc_components_benzene(benzene_modifications):
    benzene_smc = benzene_modifications['benzene']
    return {benzene_smc: benzene_smc.to_openff()}


def test_protein_component_fail(smc_components_benzene, T4_protein_component):
    errmsg = "ProteinComponents is not currently supported"
    with pytest.raises(ValueError, match=errmsg):
        interchange_packmol_creation(
            ffsettings=InterchangeFFSettings(),
            solvation_settings=PackmolSolvationSettings(),
            smc_components=smc_components_benzene,
            protein_component=T4_protein_component,
            solvent_component=None,
            solvent_offmol=None
        )


def test_get_and_set_offmol_resname(CN_molecule, caplog):
    CN_off = CN_molecule.to_openff()

    # No residue name to begin with
    assert _get_offmol_resname(CN_off) is None

    # Boop the floof
    _set_offmol_resname(CN_off, 'BOOP')

    # Does the floof accept the boop?
    assert 'BOOP' == _get_offmol_resname(CN_off)

    # Oh no, one of the atoms didn't like the boop!
    atom3 = list(CN_off.atoms)[2]
    atom3.metadata['residue_name'] = 'NOBOOP'

    with caplog.at_level(logging.WARNING):
        assert _get_offmol_resname(CN_off) is None
    assert 'Inconsistent residue name' in caplog.text

"""
1. Solvent component fails with neutralize or ion_concentration
2. No offmol is passed
3. NoCutoff settings on a vacuum system
4. Named solvent
5. Unamed solvent
  - Check we get the new residue names
  - Check we get warned about renaming
  - 
6. Named solvent with inconsistent name
7. Duplicate named smcs
8. Charged solvent
9. Uncharged solvent
10. Cube
11. Dodecahedron
12. Check we get the right residues
13. Check we get the right number of atoms
  - with a solvent w/ virtual sites
"""