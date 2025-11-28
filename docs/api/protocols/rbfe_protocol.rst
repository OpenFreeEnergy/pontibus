Hybrid Topology Relative Binding Free Energy (RBFE) Protocol
============================================================

.. _rbfe protocol api:


An extension of the :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocol` which purely
uses OpenFF for system parameterization.


Protocol API Specification
--------------------------

.. module:: pontibus.protocols.relative

.. autosummary::
   :nosignatures:
   :toctree: generated/

   HybridTopProtocol
   HybridTopProtocolResult
   HybridTopProtocolSettings
   HybridTopProtocolUnit


Protocol Settings
-----------------

.. module:: pontibus.protocols.relative.settings

.. autopydantic_model:: HybridTopProtocolSettings
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
