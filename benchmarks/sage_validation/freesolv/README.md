## Sage freesolv benchmark

Sage freesolv benchmark using pontibus.

### Inputs

An input SDF for each solute molecule with am1bccelf10 partial charges is provided under `fsolv_am1bccelf10.sdf`.

### Results

Each results file is a tab-separated values (TSV) file, which contains:
    * The solute molecule (`molecule`)
    * The pontibus calculated estimate and errors (mean and standard deviations from three repeats) under the `calc` key.
    * Experimental values and errors under the `exp` key
    * The original sage benchmark estimates and errors under the `ref` key.

The results files are:
    * `freesolv_2k4fs.tsv`: Pontibus results using 20 lambda windows, 4 fs timestep (HMR), tip3p water, 1999 solvent molecules (cubic box), 5 ns production sampling, 10.5 ns pre-alchemical equilibration, 0.5 alchemical equilibration, 2.5 ps HREX exchange frequency.
    * `freesolv_15A4fs.tsv`: Pontibus results using 14 lambda windows, 4 fs timestep (HMR), tip3p water, 15A dodecahedron solvation box, 10 ns production sampling, 10.5 ns pre-alchemical equilibration, 1 ns alchemical equilibration, 2.5 ps HREX exchange frequency.
    * `freesolv_15A2fs.tsv`: Pontibus results using 14 lambda windows, 2 fs timestep (standard mass), tip3p water, 15A dodecahedron solvation box, 10 ns production sampling, 10.5 ns pre-alchemical equilibration, 1 ns alchemical equilibration, 2.5 ps HREX exchange frequency.
