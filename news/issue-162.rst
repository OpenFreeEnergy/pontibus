**Added:**

* <news item>

**Changed:**

* In line with Interchange v0.5+, the box is no longer being scaled by 1.1
  when defining the ``number_of_solvent_molecules``. In those cases, the
  resulting box density will be much higher.
  To roughly reproduce previous behavior, the default
  ``target_density`` value has been changed to 0.715 * unit.grams / unit.mL.
  If you were defining ``number_of_solvent_molecules`` and your own
  ``target_density`` then you will need to scale the density by `(1 / (1.1^3)`
  to recover the previous behaviour.
  If you are not defining ``number_of_solvent_molecules``, ``target_density``
  should be set to 0.95 * unit.gams / unit.mL (the old default value) to
  recover previous behaviour.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
