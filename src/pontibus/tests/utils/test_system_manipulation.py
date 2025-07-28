# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openmm import System, CMMotionRemover, MonteCarloBarostat
from openmm import unit as omm_unit
from pontibus.utils.system_manipulation import (
    adjust_system,
    copy_interchange_with_replacement,
)


def test_adjust_forces_nothing():
    """
    A smoke test, this should just pass.
    """
    system = System()
    adjust_system(system)


def test_ajdust_forces_remove_com_remover():
    system = System()
    com_force = CMMotionRemover()
    system.addForce(com_force)
    adjust_system(system, remove_force_types=CMMotionRemover)

    assert system.getNumForces() == 0


def test_adjust_forces_add_comm_and_barostat():
    system = System()
    barostat = MonteCarloBarostat(1.0 * omm_unit.bar, 298.15 * omm_unit.kelvin)
    com_force = CMMotionRemover()
    adjust_system(system, add_forces=[barostat, com_force])

    assert system.getNumForces() == 2
