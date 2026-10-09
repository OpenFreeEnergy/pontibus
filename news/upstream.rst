**Added:**

* <news item>

**Changed:**

* Pontibus behaviour has been updated to match the new openfe v1.13
  behaviour (`PR #203 <https://github.com/OpenFreeEnergy/pontibus/pull/203>`_).
* Redundant lambda windows in the vacuum leg of the ASFEProtcol have now been
  removed, reducing the default number of lambda windows to 5.
* The ASFEProtocol now controls lambda schedules using the ``vacuum_lambda_settings``
  and ``solvent_lambda_settings`` entries instead of a single ``lambda_settings``.
* The ASFEProtocol now automatically carries out a symmetric RMSD
  of the alchemical ligand for solvent simulations.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Pontibus is now compatible with openmmforcefields v0.16.0.

**Security:**

* <news item>
