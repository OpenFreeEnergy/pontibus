# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe


from gufe import SmallMoleculeComponent, SolventComponent
from openfe.utils import without_oechem_backend
from openff.units import unit

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
        solvent_molecule : SmallMoleculeComponent
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
        # RDKit and OpenEye make for different smiles
        with without_oechem_backend():
            smiles = solvent_molecule.to_openff().to_smiles()

        super().__init__(
            smiles=smiles,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
            neutralize=neutralize,
            ion_concentration=ion_concentration,
        )

    @property
    def solvent_molecule(self) -> str | SmallMoleculeComponent:
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

        return {
            "solvent_molecule": self.solvent_molecule,
            "positive_ion": self.positive_ion,
            "negative_ion": self.negative_ion,
            "ion_concentration": ion_conc,
            "neutralize": self._neutralize,
        }
