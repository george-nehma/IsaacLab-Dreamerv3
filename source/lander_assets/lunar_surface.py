"""Configuration for a simple Lunar Lander robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
import numpy as np

##
# Configuration
##

LUNAR_SURFACE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/lander_assets/moon_terrain_good.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=True,
            retain_accelerations=False,
        ),
        # mass_props=sim_utils.MassPropertiesCfg(
        #     density=0.0,
        #     mass=1.0,
        # ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), rot = (1.0, 0.0, 0.0, 0.0), # joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0, "pole_to_pendulum": 0.0}
    ),
    # actuators={
    #     "cart_actuator": ImplicitActuatorCfg(
    #         joint_names_expr=["slider_to_cart"],
    #         effort_limit=400.0,
    #         velocity_limit=100.0,
    #         stiffness=0.0,
    #         damping=10.0,
    #     ),
    #     "pole_actuator": ImplicitActuatorCfg(
    #         joint_names_expr=["cart_to_pole"], effort_limit=400.0, velocity_limit=100.0, stiffness=0.0, damping=0.0
    #     ),
    #     "pendulum_actuator": ImplicitActuatorCfg(
    #         joint_names_expr=["pole_to_pendulum"], effort_limit=400.0, velocity_limit=100.0, stiffness=0.0, damping=0.0
    #     ),
    # },
)
"""Configuration for a simple Lunar Lander robot."""