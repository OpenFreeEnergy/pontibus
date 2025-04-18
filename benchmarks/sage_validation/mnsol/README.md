## Sage MNSOL benchmark

Sage MNSOL benchmarks using pontibus.

### Inputs

* An input SDF for all solvent and solute molecule with am1bccelf10 partial charges is provided under `mnsol_am1bccelf10.sdf`.
* A CSV with the Sage paper results is provided under `sage_paper_results_mnsol_2_0_0.csv`.

### Results

Each results file is a tab-separated values (TSV) file, which contains:
    * The solute molecule smiles (`solute`)
    * The solvent molecule smiles (`solvent`)
    * The pontibus calculated estimate and errors (mean and standard deviations from three repeats) under the `calc` key.
    * Experimental values and errors under the `exp` key
    * The original sage benchmark estimates and errors under the `ref` key.

The results files are:
    * `mnsol_2k.tsv`: Pontibus results using 20 lambda windows, 4 fs timestep (HMR), 1999 solvent molecules (cubic box), 5 ns production sampling, 10.5 ns pre-alchemical equilibration, 0.5 ns alchemical equilibration, 2.5 ps HREX exchange frequency. **Note:** one repeat is missing from Nc1ccccc1 (solute) | CCCCOP(=O)(OCCCC)OCCCC (solvent)
    * `mnsol_750.tsv`: Pontibus results ussing 14 lambda windows, 4 fs timestep (HMR), 750 solvent molecules (cubic box), 10 ns production sampling, 10.5 ns pre-alchemical equilibration, 1 ns alchemical equilibration, 2.5 ps HREX exchange frequency.
