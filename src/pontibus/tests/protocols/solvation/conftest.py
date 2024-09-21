import pytest
from importlib import resources
import gzip


@pytest.fixture
def afe_solv_transformation_json() -> str:
    """
    ASFE results object as created by quickrun.

    generated with devtools/gent-serialized-results.py
    """
    d = resources.files("pontibus.tests.data.solvation_protocol")
    fname = "ASFEProtocol_json_results.gz"

    with gzip.open((d / fname).as_posix(), "r") as f:
        return f.read().decode()
