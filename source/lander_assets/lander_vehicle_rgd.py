"""Configuration for a simple Lunar Lander robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
import numpy as np

##
# Configuration
##

LUNAR_LANDER_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/lander_assets/fakelander2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            retain_accelerations=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=15000.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 245/255, 184/255), metallic = 0.8),
    ),
    debug_vis=True,
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 20.0), #joint_pos={".*joint": 0.0},
    ),
)
"""Configuration for a simple Lunar Lander robot."""