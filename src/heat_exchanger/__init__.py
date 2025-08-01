__all__ = [
    "epsilon_ntu",
    "dp_friction_only",
    "dp_tube_bank",
    "PerfectGasProperties",
]  # Makes these functions imported when import * but also important for general code working


from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    rectangular_duct_friction_factor,
    rectangular_duct_nusselt,
    tube_bank_friction_factor,
    tube_bank_nusselt_from_hagen,
    tube_bank_nusselt_number_and_friction_factor,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.hex_basic import dp_friction_only, dp_tube_bank
from heat_exchanger.fluid_properties import PerfectGasProperties
from heat_exchanger.geometry_tube_bank import (
    area_heat_transfer_bank,
    area_heat_transfer_in_tubes,
    area_frontal_bank,
    axial_involute_tube_length,
    area_free_flow_bank,
    sigma_tube_bank,
)

# the first aviation is the folder name, the second is the file name
# this represents the package name and the module

# Now we have added the functions in the fleet module to the package namespace
# This means we can now import the functions directly from the package namespace with
# the import in __init__.py. Here are the different ways to import and use functions:
#
# 1. import aviation
#    aviation.fleet.passengers_per_day()
#    aviation.passengers_per_day()
#
# 2. from aviation import fleet
#    fleet.passengers_per_day()
#
# 3. from aviation import passengers_per_day
#    passengers_per_day()
#
# 4. from aviation.fleet import passengers_per_day
#    passengers_per_day()
