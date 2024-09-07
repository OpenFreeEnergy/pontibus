# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest
from openff.toolkit import Molecule
from openff.units import unit
from gufe import SmallMoleculeComponent
from pontibus.components.extended_solvent_component import (
    ExtendedSolventComponent,
)
