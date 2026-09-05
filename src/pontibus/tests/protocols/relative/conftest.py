import gzip
from importlib import resources

import pytest


@pytest.fixture
def rfe_solv_transformation_json() -> str:
    """
    HybridTop results object as created by quickrun.

    generated with devtools/gen-serialized-results.py
    """
    d = resources.files("pontibus.tests.data.relative_protocol")
    file = d / "HybridTopProtocol_solvent_json_results.gz"

    with gzip.open(file.as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore


@pytest.fixture
def rfe_vacuum_transformation_json() -> str:
    """
    Hybrid results object as created by quickrun.

    generated with devtools/gen-serialized-results.py
    """
    d = resources.files("pontibus.tests.data.relative_protocol")
    file = d / "HybridTopProtocol_vacuum_json_results.gz"

    with gzip.open(file.as_posix(), "r") as f:  # type: ignore
        return f.read().decode()  # type: ignore
