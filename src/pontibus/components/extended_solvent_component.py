from typing import Union
from openff.toolkit import Molecule
from openff.units import unit
from gufe import (
    SolventComponent,
    SmallMoleculeComponent,
)


WATER = SmallMoleculeComponent.from_openff(
    Molecule.from_smiles('O')
)


class EnhancedSolventComponent(SolventComponent):
    _solvent: SmallMoleculeComponent

    def __init__(self, *,  # force kwarg usage
                 solvent: SmallMoleculeComponent = WATER,
                 positive_ion: str = 'Na+',
                 negative_ion: str = 'Cl-',
                 neutralize: bool = True,
                 ion_concentration: unit.Quantity = 0.15 * unit.molar):
        self._solvent = solvent
        smiles = solvent.to_openff().to_smiles()
        super().__init__(
            smiles=smiles,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
            neutralize=neutralize,
            ion_concentration=ion_concentration
        )

    @property
    def solvent(self) -> Union[str, SmallMoleculeComponent]:
        """SMILES representation of the solvent molecules"""
        return self._solvent

    @classmethod
    def _from_dict(cls, d):
        """Deserialize from dict representation"""
        ion_conc = d['ion_concentration']
        d['ion_concentration'] = unit.parse_expression(ion_conc)

        return cls(**d)

    def _to_dict(self):
        """For serialization"""
        ion_conc = str(self.ion_concentration)

        if isinstance(self.solvent, SmallMoleculeComponent):
            solvent = self.solvent.to_dict()
        else:
            solvent = self.solvent

        return {
            'solvent': solvent,
            'smiles': self.smiles,
            'positive_ion': self.positive_ion,
            'negative_ion': self.negative_ion,
            'ion_concentration': ion_conc,
            'neutralize': self._neutralize
        }
