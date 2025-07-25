"""
Dev script to generate some result jsons that are used for testing

Generates
- ASFEProtocol_json_results.gz
"""

import gzip
import json
import logging
import pathlib
import tempfile

import gufe
import openfe
from gufe.tokenization import JSON_HANDLER
from openff.toolkit import AmberToolsToolkitWrapper, Molecule, RDKitToolkitWrapper
from openff.toolkit.utils.toolkit_registry import (
    ToolkitRegistry,
    toolkit_registry_manager,
)
from openff.units import unit

from pontibus.components import ExtendedSolventComponent
from pontibus.protocols.solvation import ASFEProtocol

logger = logging.getLogger(__name__)

LIGA = "[H]C([H])([H])C([H])([H])C(=O)C([H])([H])C([H])([H])[H]"

amber_rdkit = ToolkitRegistry([RDKitToolkitWrapper(), AmberToolsToolkitWrapper()])


def get_molecule(smi, name):
    with toolkit_registry_manager(amber_rdkit):
        m = Molecule.from_smiles(smi)
        m.generate_conformers()
        m.assign_partial_charges(partial_charge_method="am1bcc")
    return openfe.SmallMoleculeComponent.from_openff(m, name=name)


def execute_and_serialize(dag, protocol, simname):
    logger.info(f"running {simname}")
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = pathlib.Path(tmpdir)
        dagres = gufe.protocols.execute_DAG(
            dag,
            shared_basedir=workdir,
            scratch_basedir=workdir,
            keep_shared=False,
            n_retries=3,
        )
    protres = protocol.gather([dagres])

    outdict = {
        "estimate": protres.get_estimate(),
        "uncertainty": protres.get_uncertainty(),
        "protocol_result": protres.to_dict(),
        "unit_results": {unit.key: unit.to_keyed_dict() for unit in dagres.protocol_unit_results},
    }

    with gzip.open(f"{simname}_json_results.gz", "wt") as zipfile:
        json.dump(outdict, zipfile, cls=JSON_HANDLER.encoder)


def generate_ahfe_settings():
    settings = ASFEProtocol.default_settings()
    settings.solvent_equil_simulation_settings.equilibration_length_nvt = 10 * unit.picosecond
    settings.solvent_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.solvent_equil_simulation_settings.production_length = 10 * unit.picosecond
    settings.solvent_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.solvent_simulation_settings.production_length = 500 * unit.picosecond
    settings.vacuum_equil_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.vacuum_equil_simulation_settings.production_length = 10 * unit.picosecond
    settings.vacuum_simulation_settings.equilibration_length = 10 * unit.picosecond
    settings.vacuum_simulation_settings.production_length = 500 * unit.picosecond
    settings.protocol_repeats = 3
    settings.vacuum_engine_settings.compute_platform = "CPU"
    settings.solvent_engine_settings.compute_platform = "CUDA"

    return settings


def generate_asfe_json_water(smc):
    settings = generate_ahfe_settings()
    settings.solvation_settings.assign_solvent_charges = False
    protocol = ASFEProtocol(settings=settings)
    sysA = openfe.ChemicalSystem({"ligand": smc, "solvent": ExtendedSolventComponent()})
    sysB = openfe.ChemicalSystem({"solvent": ExtendedSolventComponent()})

    dag = protocol.create(stateA=sysA, stateB=sysB, mapping=None)

    execute_and_serialize(dag, protocol, "ASFEProtocol_water")


def generate_asfe_json_octanol(smc):
    settings = generate_ahfe_settings()
    settings.solvation_settings.assign_solvent_charges = True
    protocol = ASFEProtocol(settings=settings)
    solvent = Molecule.from_smiles("CCCCCCCCO")
    solvent.assign_partial_charges(partial_charge_method="am1bcc")
    solvent.generate_conformers(n_conformers=1)
    solvent_component = ExtendedSolventComponent(
        solvent_molecule=openfe.SmallMoleculeComponent.from_openff(solvent),
    )
    sysA = openfe.ChemicalSystem({"ligand": smc, "solvent": solvent_component})
    sysB = openfe.ChemicalSystem({"solvent": solvent_component})

    dag = protocol.create(stateA=sysA, stateB=sysB, mapping=None)

    execute_and_serialize(dag, protocol, "ASFEProtocol_octanol")


if __name__ == "__main__":
    molA = get_molecule(LIGA, "ligandA")
    # generate_asfe_json_water(molA)
    generate_asfe_json_octanol(molA)
