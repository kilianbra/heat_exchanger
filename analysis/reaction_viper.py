import sys
import os
# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from heat_exchanger.hex_basic import ntu, dp_tube_bank, dp_friction_only
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.geometry_tube_bank import (
    area_heat_transfer_bank,
    area_heat_transfer_in_tubes,
    area_frontal_bank,
    area_free_flow_bank,
    area_free_flow_in_tubes,
    axial_involute_tube_length,
    sigma_tube_bank,
)
from heat_exchanger.fluid_properties import PerfectGasProperties
from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    tube_bank_nusselt_number_and_friction_factor,
)