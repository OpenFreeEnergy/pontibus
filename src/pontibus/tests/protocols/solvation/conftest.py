import gzip
from importlib import resources

import gufe
import pytest


@pytest.fixture
def charged_benzene(benzene_modifications):
    benzene_offmol = benzene_modifications["benzene"].to_openff()
    benzene_offmol.assign_partial_charges(partial_charge_method="gasteiger")
    return gufe.SmallMoleculeComponent.from_openff(benzene_offmol)


@pytest.fixture
def afe_solv_water_transformation_json() -> str:
    """
    ASFE results object as created by quickrun.

    generated with devtools/gent-serialized-results.py
    """
    d = resources.files("pontibus.tests.data.solvation_protocol")
    fname = "ASFEProtocol_water_json_results.gz"

    with gzip.open((d / fname).as_posix(), "r") as f:
        return f.read().decode()


@pytest.fixture
def afe_solv_octanol_transformation_json() -> str:
    """
    ASFE results object as created by quickrun.

    generated with devtools/gent-serialized-results.py
    """
    d = resources.files("pontibus.tests.data.solvation_protocol")
    fname = "ASFEProtocol_octanol_json_results.gz"

    with gzip.open((d / fname).as_posix(), "r") as f:
        return f.read().decode()
