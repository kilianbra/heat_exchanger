__all__ = [
    "epsilon_ntu",
    "dp",
]  # Makes these functions imported when import * but also important for general code working


from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    rectangular_duct_friction_factor,
    rectangular_duct_nusselt,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.hex_basic import dp
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
