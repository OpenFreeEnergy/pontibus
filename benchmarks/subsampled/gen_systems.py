import gzip
import json
import pathlib

from gufe import ChemicalSystem, SmallMoleculeComponent
from gufe.tokenization import JSON_HANDLER
from openff.toolkit import Molecule
from pontibus.components import ExtendedSolventComponent
from rdkit import Chem


def add_chemical_systems(
    sdffile: str,
    dataset_name: str,
    solvents: dict[str, SmallMoleculeComponent],
    systems: list[ChemicalSystem],
) -> None:
    """
    Add Solute + Solvent ChemicalSystems to running list.

    Parameters
    ----------
    sdffile : str
      The SDF file to read entries from.
    dataset_name : str
      The name of the dataset.
    solvents: dict[str, SmallMoleculeComponent]
      Running dictionary of solvents to draw & store prepared solvent
      molecules from/to.
    systems: list[ChemicalSystem]
      Runing list of ChemicalSystems we are appending to.
    """
    for i, rdmol in enumerate(Chem.SDMolSupplier(sdffile, removeHs=False)):
        offmol = Molecule.from_rdkit(rdmol)
        offmol.assign_partial_charges(partial_charge_method="am1bccelf10")
        solvent_smi = rdmol.GetProp("solvent")
        if solvent_smi not in solvents.keys():
            solvent_offmol = Molecule.from_smiles(solvent_smi)
            solvent_offmol.generate_conformers(n_conformers=1)
            solvent_offmol.assign_partial_charges(partial_charge_method="am1bccelf10")
            solvents[solvent_smi] = SmallMoleculeComponent.from_openff(solvent_offmol)

        systems.append(
            ChemicalSystem(
                {
                    "solute": SmallMoleculeComponent.from_openff(offmol),
                    "solvent": ExtendedSolventComponent(
                        solvent_molecule=solvents[solvent_smi]
                    ),
                },
                name=f"molecule{i}_{dataset_name}",
            )
        )


def store_chemical_systems(systems: list[ChemicalSystem], outdir: pathlib.Path):
    """
    Store ChemicalSystems to gzip file.

    Parameters
    ----------
    systems: list[ChemicalSystem]
      List of ChemicalSystems to store to file.
    """
    for system in systems:
        with gzip.open(outdir / f"{system.name}_chemicalsystem.gz", "wt") as zipfile:
            json.dump(system.to_dict(), zipfile, cls=JSON_HANDLER.encoder)


if __name__ == "__main__":
    solvents: dict[str, SmallMoleculeComponent] = {}
    systems: list[ChemicalSystem] = []

    add_chemical_systems("sub_sampled_fsolv.sdf", "fsolv", solvents, systems)
    add_chemical_systems("sub_sampled_mnsol.sdf", "mnsol", solvents, systems)

    outdir = pathlib.Path("chemicalsystems")
    outdir.mkdir(exist_ok=True)
    store_chemical_systems(systems, outdir)
