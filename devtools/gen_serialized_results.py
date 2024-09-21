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
from openff.toolkit import (
    Molecule, RDKitToolkitWrapper, AmberToolsToolkitWrapper
)
from openff.toolkit.utils.toolkit_registry import (
    toolkit_registry_manager, ToolkitRegistry
)
from openff.units import unit
from kartograf.atom_aligner import align_mol_shape
from kartograf import KartografAtomMapper
import gufe
from gufe.tokenization import JSON_HANDLER
import openfe
from pontibus.protocols.solvation import ASFEProtocol
from pontibus.components import ExtendedSolventComponent


logger = logging.getLogger(__name__)

LIGA = "[H]C([H])([H])C([H])([H])C(=O)C([H])([H])C([H])([H])[H]"

amber_rdkit = ToolkitRegistry(
    [RDKitToolkitWrapper(), AmberToolsToolkitWrapper()]
)


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
            n_retries=3
        )
    protres = protocol.gather([dagres])

    outdict = {
        "estimate": protres.get_estimate(),
        "uncertainty": protres.get_uncertainty(),
        "protocol_result": protres.to_dict(),
        "unit_results": {
            unit.key: unit.to_keyed_dict()
            for unit in dagres.protocol_unit_results
        }
    }

    with gzip.open(f"{simname}_json_results.gz", 'wt') as zipfile:
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
    settings.vacuum_engine_settings.compute_platform = 'CPU'
    settings.solvent_engine_settings.compute_platform = 'CUDA'

    return settings

    
def generate_asfe_json(smc):
    protocol = ASFEProtocol(settings=generate_ahfe_settings())
    sysA = openfe.ChemicalSystem(
        {"ligand": smc, "solvent": ExtendedSolventComponent()}
    )
    sysB = openfe.ChemicalSystem(
        {"solvent": ExtendedSolventComponent()}
    )

    dag = protocol.create(stateA=sysA, stateB=sysB, mapping=None)

    execute_and_serialize(dag, protocol, "ASFEProtocol")


if __name__ == "__main__":
    molA = get_molecule(LIGA, "ligandA")
    generate_asfe_json(molA)
