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
    file = d / "ASFEProtocol_water_json_results.gz"

    with gzip.open(file.as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def afe_solv_octanol_transformation_json() -> str:
    """
    ASFE results object as created by quickrun.

    generated with devtools/gent-serialized-results.py
    """
    d = resources.files("pontibus.tests.data.solvation_protocol")
    file = d / "ASFEProtocol_octanol_json_results.gz"

    with gzip.open(file.as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore
