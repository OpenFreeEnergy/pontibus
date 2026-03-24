**Added:**

* <news item>

**Changed:**

* In line with Interchange v0.5+, the box is no longer being scaled by 1.1
  when defining the ``number_of_solvent_molecules``. In those cases, the
  resulting box density will be much higher.
  If you were defining ``number_of_solvent_molecules`` and then you will
  need to scale the density by `(1 / (1.1^3)` instead of using the
  default `target_density` (currently default of 0.95 * unit.grams / unit.mL).
  In practice this means setting `target_density` to roughly
  0.715 unit.grams / unit.mL when setting `number_of_solvent_molecules`.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
