import gzip
from importlib import resources

import pytest


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
