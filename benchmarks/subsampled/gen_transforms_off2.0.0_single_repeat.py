import gzip
import json
import pathlib

from gufe import AlchemicalNetwork, ChemicalSystem, Transformation
from gufe.tokenization import JSON_HANDLER
from openff.toolkit import Molecule
from openff.units import unit
from pontibus.protocols.solvation import ASFEProtocol
from pontibus.protocols.solvation.settings import PackmolSolvationSettings


def deserialize_system(file: pathlib.Path):
    with gzip.open(file, "r") as f:
        d = json.loads(f.read().decode(), cls=JSON_HANDLER.decoder)
    return ChemicalSystem.from_dict(d)


def get_water_settings():
    settings = ASFEProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.solvent_forcefield_settings.forcefields = [
        "openff-2.0.0.offxml",
        "tip3p.offxml",
    ]
    # settings.solvent_forcefield_settings.hydrogen_mass = 1.007947
    settings.vacuum_forcefield_settings.forcefields = [
        "openff-2.0.0.offxml",
    ]
    # settings.vacuum_forcefield_settings.hydrogen_mass = 1.007947
    settings.solvent_simulation_settings.time_per_iteration = 5 * unit.picosecond
    settings.vacuum_simulation_settings.time_per_iteration = 5 * unit.picosecond
    settings.vacuum_engine_settings.compute_platform = "CPU"
    settings.solvent_engine_settings.compute_platform = "CUDA"
    settings.solvation_settings = PackmolSolvationSettings(
        number_of_solvent_molecules=1999,
        box_shape="cube",
        assign_solvent_charges=False,
        solvent_padding=None,
    )
    # settings.integrator_settings.timestep = 2 * unit.femtosecond
    return settings


def get_nonwater_settings():
    settings = ASFEProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.solvent_forcefield_settings.forcefields = [
        "openff-2.0.0.offxml",
    ]
    # settings.solvent_forcefield_settings.hydrogen_mass = 1.007947
    settings.vacuum_forcefield_settings.forcefields = [
        "openff-2.0.0.offxml",
    ]
    # settings.vacuum_forcefield_settings.hydrogen_mass = 1.007947
    settings.solvent_simulation_settings.time_per_iteration = 5 * unit.picosecond
    settings.vacuum_simulation_settings.time_per_iteration = 5 * unit.picosecond
    settings.vacuum_engine_settings.compute_platform = "CPU"
    settings.solvent_engine_settings.compute_platform = "CUDA"
    settings.solvation_settings = PackmolSolvationSettings(
        number_of_solvent_molecules=1999,
        box_shape="cube",
        assign_solvent_charges=True,
        solvent_padding=None,
    )
    # settings.integrator_settings.timestep = 2 * unit.femtosecond
    return settings


def get_transformation(stateA):
    solvent = stateA.components["solvent"]

    water = Molecule.from_smiles("O")
    if water.is_isomorphic_with(solvent.solvent_molecule.to_openff()):
        settings = get_water_settings()
    else:
        settings = get_nonwater_settings()

    stateB = ChemicalSystem({"solvent": solvent})
    protocol = ASFEProtocol(settings=settings)
    return Transformation(
        stateA=stateA, stateB=stateB, mapping=None, protocol=protocol, name=stateA.name
    )


def run(outdir: pathlib.Path):
    systems: list[ChemicalSystem] = []

    systems_dir = pathlib.Path("chemicalsystems")
    system_files = systems_dir.glob("*.gz")

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
    outdir = pathlib.Path("off2.0.0_single_repeat_inputs")
    outdir.mkdir(exist_ok=False)
    run(outdir)
