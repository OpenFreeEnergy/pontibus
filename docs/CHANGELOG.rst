===================
pontibus Change Log
===================

.. current developments

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
