===================
pontibus Change Log
===================

.. current developments

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
