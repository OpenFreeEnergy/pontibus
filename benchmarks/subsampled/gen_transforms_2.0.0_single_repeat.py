import pathlib
import json
import gzip
from openff.units import unit
from gufe.tokenization import JSON_HANDLER
from gufe import (
    AlchemicalNetwork,
    ChemicalSystem,
    Transformation
)
from pontibus.protocols.solvation import ASFEProtocol


def deserialize_system(file: pathlib.Path):
    with gzip.open(file, 'r') as f:
        d = json.loads(
            f.read().decode(),
            cls=JSON_HANDLER.decoder
        )
    return ChemicalSystem.from_dict(d)


def get_settings():
    settings = ASFEProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.solvent_forcefield_settings.forcefields = [
        "openff-2.0.0.offxml",
        "tip3p.offxml",
    ]
    settings.vacuum_forcefield_settings.forcefields = [
        "openff-2.0.0.offxml",
    ]
    settings.solvent_simulation_settings.time_per_iteration = 5 * unit.picosecond
    settings.vacuum_simulation_settings.time_per_iteration = 5 * unit.picosecond
    return settings


def get_transformation(stateA):
    settings = get_settings()
    stateB = stateA.components['solvent']
    protocol = ASFEProtocol(settings=settings)
    return Transformation(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
        protocol=protocol,
        name=stateA.name
    )


def run(outdir: pathlib.Path):
    systems: list[ChemicalSystem] = []

    systems_dir = pathlib.Path('chemicalsystems')
    system_files = systems_dir.glob('*.gz')

    for file in system_files:
        systems.append(deserialize_system(file))

    transformations = []
    for system in systems:
        transformations.append(get_transformation(system))

    alchemical_network = AlchemicalNetwork(transformations)
    an_outfile = outdir / "alchemical_network.json"
    json.dump(
        alchemical_network.to_dict(),
        an_outfile.open(mode="w"),
        cls=JSON_HANDLER.encoder,
    )

    transforms_dir = outdir / "transformations"
    transforms_dir.mkdir(exist_ok=True, parents=True)

    for transform in alchemical_network.edges:
        transform.dump(transforms_dir / f"{transform.name}.json")


if __name__ == "__main__":
    outdir = pathlib.Path('2.0.0_single_repeat_inputs')
    outdir.mkdir(exist_ok=False)
    run(outdir)