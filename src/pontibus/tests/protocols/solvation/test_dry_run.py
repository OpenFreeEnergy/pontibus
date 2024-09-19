# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from gufe import ChemicalSystem
from pontibus.protocols.solvation import ASFEProtocol, ASFESolventUnit, ASFEVacuumUnit
from pontibus.components import ExtendedSolventComponent


@pytest.mark.parametrize("method", ["repex", "sams", "independent", "InDePeNdENT"])
def test_dry_run_vacuum_benzene(benzene_modifications, method, tmpdir):
    s = ASFEProtocol.default_settings()
    s.protocol_repeats = 1
    s.vacuum_simulation_settings.sampler_method = method

    protocol = ASFEProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "solvent": ExtendedSolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "solvent": ExtendedSolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    vac_unit = [u for u in prot_units if isinstance(u, ASFEVacuumUnit)]
    sol_unit = [u for u in prot_units if isinstance(u, ASFESolventUnit)]

    assert len(vac_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        vac_sampler = vac_unit[0].run(dry=True)["debug"]["sampler"]
        assert not vac_sampler.is_periodic
