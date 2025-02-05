Absolute Solvation Free Energy (ASFE) Protocol
==============================================

.. _asfe protocol api:

A Protocol for running arbitrary solvent solvation free energy calculations.


Protocol API Specification
--------------------------

.. module:: pontibus.protocols.solvation

.. autosummary::
   :nosignatures:
   :toctree: generated/

   ASFEProtocol
   ASFEProtocolResult
   ASFESettings
   ASFESolventUnit
   ASFEVacuumUnit


Protocol Settings
-----------------

.. module:: pontibus.protocols.solvation.settings

.. autopydantic_model:: ASFESettings
   :model-show-json: False
   :model-show-field-summary: False
   :model-show-config-member: False
   :model-show-config-summary: False
   :model-show-validator-members: False
   :model-show-validator-summary: False
   :field-list-validators: False
   :inherited-members: SettingsBaseModel
   :exclude-members: get_defaults
   :member-order: bysource
