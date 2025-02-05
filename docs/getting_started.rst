Getting Started
===============

This page details how to get started with pontibus.


Developer Install
*****************

To install the development version of the pontibus, you should do a source
installation in the following manner::

    git clone https://github.com/OpenFreeEnergy/pontibus.git

    cd pontibus
    mamba env create -f environment.yml

    mamba activate pontibus
    python -m pip install -e .
