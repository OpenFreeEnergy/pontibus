# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import importlib
import os
from importlib import resources

import gufe
import pytest
from gufe import SmallMoleculeComponent
from rdkit import Chem


class SlowTests:
    """Plugin for handling fixtures that skips slow tests

    Fixtures
    --------

    Currently three fixture types are handled:
      * `gpu`:
        GPU tests that are meant to be run to truly put the code
        through a real run.

      * `slow`:
        Unit tests that just take too long to be running regularly.

      * `cpuvslow`:
        Unit tests that run very slowly and likely require high
        performance CPU runners.


    How to use the fixtures
    -----------------------

    To add these fixtures simply add a `@pytest.mark.gpu` or
    `@pytest.mark.slow` or `@pytest.mark.cpuslow`
    decorator to the relevant function or class.


    How to run tests marked by these fixtures
    -----------------------------------------

    To run the `gpu` tests, either use the `--gpu` flag
    when invoking pytest, or set the environment variable
    `PONTIBUS_GPU_TESTS` to `true`. Note: triggering `gpu` will
    automatically also trigger tests marked as `slow`.

    To run the `slow` tests, either use the `--slow` flag when invoking
    pytest, or set the environment variable `PONTIBUS_SLOW_TESTS` to `true`.

    To run the `cpuvslow` tests, either use the `--cpuvslow` flag when
    invoking pytest, or set the environment variable `PONTIBUS_CPUVSLOW_TESTS`
    to `true`.
    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _modify_slow(items, config):
        msg = (
            "needs --slow pytest cli option or the environment variable "
            "`PONTIBUS_SLOW_TESTS` set to `True` to run"
        )
        skip_slow = pytest.mark.skip(reason=msg)
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    @staticmethod
    def _modify_gpu(items, config):
        msg = (
            "needs --gpu pytest cli option or the environment "
            "variable `PONTIBUS_GPU_TESTS` set to `True` to run"
        )
        skip_int = pytest.mark.skip(reason=msg)
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_int)

    @staticmethod
    def _modify_cpuvslow(items, config):
        msg = (
            "needs --cpuvslow pytest cli option or the environment "
            "variable `PONTIBUS_CPUVSLOW_TESTS` set to `True` to run"
        )
        skip_int = pytest.mark.skip(reason=msg)
        for item in items:
            if "cpuvslow" in item.keywords:
                item.add_marker(skip_int)

    def pytest_collection_modifyitems(self, items, config):
        if (
            config.getoption("--gpu")
            or os.getenv("PONTIBUS_GPU_TESTS", default="false").lower() == "true"
        ):
            self._modify_cpuvslow(items, config)
        elif (
            config.getoption("--cpuvslow")
            or os.getenv("PONTIBUS_CPUVSLOW_TESTS", default="false").lower() == "true"
        ):
            self._modify_gpu(items, config)
        elif (
            config.getoption("--slow")
            or os.getenv("PONTIBUS_SLOW_TESTS", default="false").lower() == "true"
        ):
            self._modify_gpu(items, config)
            self._modify_cpuvslow(items, config)
        else:
            self._modify_gpu(items, config)
            self._modify_cpuvslow(items, config)
            self._modify_slow(items, config)


# allow for optional slow tests
# See: https://docs.pytest.org/en/latest/example/simple.html
def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="run GPU tests",
    )
    parser.addoption(
        "--cpuvslow",
        action="store_true",
        default=False,
        help="run very slow CPU tests"
    )


def pytest_configure(config):
    config.pluginmanager.register(SlowTests(config), "slow")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as long integration test")
    config.addinivalue_line("markers", "cpuvslow: mark tests as CPU-only and very slow")


@pytest.fixture(scope="session")
def benzene_modifications():
    files = {}
    with importlib.resources.files("openfe.tests.data") as d:
        fn = str(d / "benzene_modifications.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture()
def CN_molecule():
    """
    A basic CH3NH2 molecule for quick testing.
    """
    with resources.files("openfe.tests.data") as d:
        fn = str(d / "CN.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)

        smc = [SmallMoleculeComponent(i) for i in supp][0]

    return smc


@pytest.fixture(scope="session")
def T4_protein_component():
    with resources.files("openfe.tests.data") as d:
        fn = str(d / "181l_only.pdb")
        comp = gufe.ProteinComponent.from_pdb_file(fn, name="T4_protein")

    return comp
