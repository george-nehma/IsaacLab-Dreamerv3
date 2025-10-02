"""Configuration for a simple Lunar Lander robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
import numpy as np

##
# Configuration
##

# LUNAR_LANDER_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"source/lander_assets/landernew.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             max_angular_velocity=1000.0,
#             max_linear_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#             disable_gravity=False,
#             retain_accelerations=False,
#         ),
#         # mass_props=sim_utils.MassPropertiesCfg(
#         #     density=0.0,
#         #     mass=1.0,
#         # ),
#         # collision_props=sim_utils.CollisionPropertiesCfg(
#         #     collision_enabled=True,
#         # ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
#         ),
        
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 20.0), joint_pos={".*Joint:0": 0.0},
#     ),
#     actuators={
#         "pad_actuator": ImplicitActuatorCfg(
#             joint_names_expr=[".*Joint:0"],
#             effort_limit_sim=400.0,
#             velocity_limit_sim=100.0,
#             stiffness=0.0,
#             damping=10.0,
#         ),
#     },
# )


LUNAR_LANDER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/lander_assets/landernew.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 20.0),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for a simple Lunar Lander robot."""