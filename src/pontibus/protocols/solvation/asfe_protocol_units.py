# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.protocols.openmm_afe.ahfe_units import (
    SolventComponentsMixin,
    SolventSettingsMixin,
    VacuumComponentsMixin,
    VacuumSettingsMixin,
)
from openfe.protocols.openmm_afe.base_afe_units import (
    BaseAbsoluteMultiStateAnalysisUnit,
    BaseAbsoluteMultiStateSimulationUnit,
)

from pontibus.protocols.solvation.base import BaseASFEUnit


class ASFEVacuumSetupUnit(VacuumComponentsMixin, VacuumSettingsMixin, BaseASFEUnit):
    """
    Protocol Unit for setting up the vacuum phase of an absolute solvation
    free energy transformation.
    """

    simtype: str = "vacuum"


class ASFEVacuumSimUnit(
    VacuumComponentsMixin, VacuumSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    Protocol Unit for running the vacuum phase multistate simulation of an
    absolute solvation free energy transformation.
    """

    simtype: str = "vacuum"


class ASFEVacuumAnalysisUnit(VacuumSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    Protocol Unit for analysing the vacuum phase of an absolute solvation
    free energy transformation.
    """

    simtype: str = "vacuum"


class ASFESolventSetupUnit(SolventComponentsMixin, SolventSettingsMixin, BaseASFEUnit):
    """
    Protocol Unit for setting up the solvent phase of an absolute solvation
    free energy transformation.
    """

    simtype: str = "solvent"


class ASFESolventSimUnit(
    SolventComponentsMixin, SolventSettingsMixin, BaseAbsoluteMultiStateSimulationUnit
):
    """
    Protocol Unit for running the solvent phase multistate simulation of an
    absolute solvation free energy transformation.
    """

    simtype: str = "solvent"


class ASFESolventAnalysisUnit(SolventSettingsMixin, BaseAbsoluteMultiStateAnalysisUnit):
    """
    Protocol Unit for analysing the solvent phase of an absolute solvation
    free energy transformation.
    """

    simtype: str = "solvent"
