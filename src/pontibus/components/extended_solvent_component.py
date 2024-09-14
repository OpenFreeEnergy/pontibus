# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Union
from openff.units import unit
from gufe import (
    SolventComponent,
    SmallMoleculeComponent,
)
from gufe.tokenization import (
    modify_dependencies,
    TOKENIZABLE_CLASS_REGISTRY,
    TOKENIZABLE_REGISTRY,
    GufeKey,
    is_gufe_key_dict,
    _from_dict,
    GufeTokenizable,
)
from pontibus.utils.molecules import WATER


class ExtendedSolventComponent(SolventComponent):
    _solvent_molecule: SmallMoleculeComponent

    def __init__(
        self,
        *,  # force kwarg usage
        solvent_molecule: SmallMoleculeComponent = WATER,
        positive_ion: str = "Na+",
        negative_ion: str = "Cl-",
        neutralize: bool = False,
        ion_concentration: unit.Quantity = 0.0 * unit.molar,
    ):
        """
        Parameters
        ----------
        solvent_molecule : SmallMoleculeComponent, optional
          SmallMoleculeComponent defining the solvent, default
          is a water molecule.
        positive_ion, negative_ion : str
          the pair of ions which is used to neutralize (if neutralize=True) and
          bring the solvent to the required ionic concentration.  Must be a
          positive and negative monoatomic ions, defaults "Na+", "Cl-"
        neutralize : bool, optional
          if the net charge on the chemical state is neutralized by the ions in
          this solvent component.  Default `True`
        ion_concentration : openff-units.unit.Quantity, optional
          ionic concentration required, default 0.15 * unit.molar
          this must be supplied with units, e.g. "1.5 * unit.molar"

        Examples
        --------
        To create a sodium chloride solution at 0.2 molar concentration::

          >>> s = SolventComponent(position_ion='Na', negative_ion='Cl',
          ...                      ion_concentration=0.2 * unit.molar)

        To create a methane solvent::

          >>> METHANE = SmallMoleculeComponent.from_openff(
          ...    Molecule.from_smiles('C')
          ... )
          >>> s = SolventComponent(solvent_molecule=METHANE)

        """
        self._solvent_molecule = solvent_molecule
        smiles = solvent_molecule.to_openff().to_smiles()
        super().__init__(
            smiles=smiles,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
            neutralize=neutralize,
            ion_concentration=ion_concentration,
        )

    @property
    def solvent_molecule(self) -> Union[str, SmallMoleculeComponent]:
        """SmallMoleculeComponent representation of the solvent molecules"""
        return self._solvent_molecule

    @classmethod
    def _from_dict(cls, d):
        """Deserialize from dict representation"""
        ion_conc = d["ion_concentration"]
        d["ion_concentration"] = unit.parse_expression(ion_conc)

        return cls(**d)

    def _to_dict(self):
        """For serialization"""
        ion_conc = str(self.ion_concentration)

        if isinstance(self.solvent_molecule, SmallMoleculeComponent):
            solvent = self.solvent_molecule.to_dict()
        else:
            solvent = self.solvent_molecule

        return {
            "solvent_molecule": solvent,
            "positive_ion": self.positive_ion,
            "negative_ion": self.negative_ion,
            "ion_concentration": ion_conc,
            "neutralize": self._neutralize,
        }

    @classmethod
    def from_keyed_dict(cls, dct: dict):
        """Generate an instance from keyed dict representation.

        Parameters
        ----------
        dct : Dict
            A dictionary produced by `to_keyed_dict` to instantiate from.
            If an identical instance already exists in memory, it will be
            returned.  Otherwise, a new instance will be returned.

        Returns
        -------
        obj : ExtendedSolventComponent
            An object instance constructed from the input keyed dictionary.

        Notes
        -----
        This method is re-implemented in the ExtendedSolventComponent subclass
        due to gufe tokenization not working as intended with GufeTokenizables
        containing other GufeTokenizables.

        """
        registry = TOKENIZABLE_CLASS_REGISTRY
        dct = modify_dependencies(
            dct,
            lambda d: registry[GufeKey(d[":gufe-key:"])],
            is_gufe_key_dict,
            mode="decode",
            top=True,
        )

        return from_dict_depth_one(dct)

    @classmethod
    def from_shallow_dict(cls, dct: dict):
        """Generate an instance from shallow dict representation.

        Parameters
        ----------
        dct : Dict
            A dictionary produced by `to_shallow_dict` to instantiate from.
            If an identical instance already exists in memory, it will be
            returned.  Otherwise, a new instance will be returned.

        Returns
        -------
        obj : ExtendedSolventComponent
            An object instance constructed from the input shallow dictionary.

        Notes
        -----
        This method is re-implemented in the ExtendedSolventComponent subclass
        due to gufe tokenization not working as intended with GufeTokenizables
        containing other GufeTokenizables.

        """
        return from_dict_depth_one(dct)


def from_dict_depth_one(dct: dict) -> GufeTokenizable:
    obj = _from_dict_depth_one(dct)
    # When __new__ is called to create ``obj``, it should be added to the
    # TOKENIZABLE_REGISTRY. However, there seems to be some case (race
    # condition?) where this doesn't happen, leading to a KeyError inside
    # the dictionary if we use []. (When you drop into PDB and run the same
    # line that gave the error, you get the object back.) With ``get``,
    # ``thing`` becomes None, which is also what it would be if the weakref
    # was to a deleted object.
    thing = TOKENIZABLE_REGISTRY.get(obj.key)

    if thing is None:  # -no-cov-
        return obj
    else:
        return thing


def _from_dict_depth_one(dct: dict) -> GufeTokenizable:
    """
    Helper method to enable a from_dict that also
    deserializes attributes that are GufeTokenizables.
    """

    new_dct = {}

    for entry in dct:
        if isinstance(dct[entry], dict) and "__qualname__" in dct[entry]:
            new_dct[entry] = _from_dict(dct[entry])
        else:
            new_dct[entry] = dct[entry]

    return _from_dict(new_dct)
