# Welcome to  "Heat Exchanger Model"

A simple heat exchanger model.

The goal of this package is to help heat exchanger users to obtain some physical pressure drop insights in applications where volume and weight are very sensitive like aviation.

Rather than setting an allowable pressure drop and then deriving the frontal area required, an available frontal area is inputted and then an effectiveness/pressure drop curve is outputed.

## Core Documentation

Click [here](./0Dmodel.md) to view more details about the basic 0D heat exchanger model

Click [here](./eps_ntu.md) for details about the effectiveness-NTU method and the $P_n(y)$ polynomial

## Cycle Analysis Documentation

Click [here](./cycle_tsfc_documentation.md) for comprehensive documentation of the TSFC cycle analysis script (`zeli_og.py`)

Click [here](./cycle_tsfc_differences.md) for an analysis of differences between the original and refactored TSFC cycle analysis scripts

## Geometry Documentation

Click [here](./geometry_radial_spiral.md) for documentation of the radial spiral heat exchanger geometry, including key assumptions and solver methods

Click [here](./geometry_tube_bank_straight.md) for documentation of the straight tube bank geometry, including annular and box configurations
