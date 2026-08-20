===================
pontibus Change Log
===================

.. current developments

vv0.5.0
====================

**Changed:**

* In line with Interchange v0.5+, the box is no longer being scaled by 1.1
  when defining the ``number_of_solvent_molecules``. In those cases, the
  resulting box density will be much higher.
  If you were defining ``number_of_solvent_molecules`` and then you will
  need to scale the density by `(1 / (1.1^3)` instead of using the
  default `target_density` (currently default of 0.95 * unit.grams / unit.mL).
  In practice this means setting `target_density` to roughly
  0.715 unit.grams / unit.mL when setting `number_of_solvent_molecules`.
* The ASFE and HybridTop Protocols have been updated to work
  with openfe v1.9. Notably this means that the Protocols have
  now been split into three ProtocolUnits (PR #190).



v0.4.0
====================

**Added:**

* AbsoluteAlchemicalFactory options to control LJ softcore parameters
  are now exposed in the settings and can be used with both the standard
  and experimental factory (PR #171).



v0.3.0
====================

**Added:**

* Updated API documentation (PR #166)
* ASFEProtocol now has a fully implemented `validate` method,
  you can now call these ahead of Transformation execution
  to check that input parameters work with the Protocol (PR #163).

**Fixed:**

* Pontibus is now compatible with new GufeQuantity
  changes introduced with gufe v1.7 (PR #163).



v0.2.0
====================

**Added:**

* The ability to solvate systems using OpenMM for the RFE Protocol (PR #140).

**Changed:**

* The default solvation backend for RFE calculations is now OpenMM and
  the target density is 0.75 g/L.



v0.1
====================

**Added:**

* Ability to neutralize systems when solvated with water (Issue #111).
* A new experimental HybridTop Protocol (issue #73)

**Fixed:**

* Fixes missing barostat in non-alchemical simulations for SFE Protocol (issue #114)



v0.0.2
====================

**Changed:**

* Protocol now actively removes CMMotionRemover forces if present.
* Compatibility with gufe 1.3 (added correct settings_cls definition).
* Switched from versioningit to setuptools-scm



v0.0.1
====================

**Added:**

* Initial release of pontibus. This includes an ASFE Protocol
  which supports arbitrary solvent systems that are prepared
  solely with the OpenFF stack.
