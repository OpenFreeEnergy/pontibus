**Added:**

* ``InterchangeFFSettings`` now has a ``small_molecule_forcefield`` option,
  matching openfe's ``RelativeHybridTopologyProtocol``. When set, all
  ``SmallMoleculeComponent`` molecules are parameterized with this SMIRNOFF
  force field (e.g. ``"openff-2.2.1"``, normalized to ``"openff-2.2.1.offxml"``),
  separately from the solvent (including non-water solvents), ions and
  protein, which use ``forcefields``. Only SMIRNOFF (``.offxml``) force
  fields are supported. Defaults to ``None``, keeping the previous behavior.

**Changed:**

* When ``small_molecule_forcefield`` is set, small molecules are placed
  after all other molecules in the system.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
