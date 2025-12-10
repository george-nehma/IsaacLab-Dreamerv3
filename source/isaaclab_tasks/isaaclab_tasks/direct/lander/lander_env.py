# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math
from isaaclab.sensors import Camera, CameraCfg, save_images_to_file, ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, Imu, ImuCfg, patterns
import isaacsim.core.utils.numpy.rotations as rot_utils

##
# Pre-defined configs
##

import os
import sys

isaaclab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, isaaclab_root)

from source.lander_assets.lander_vehicle_rgd import LUNAR_LANDER_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class LanderEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: LanderEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class LanderEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 12
    episode_length_s = 90.0
    debug_vis = True
    # action_scale = 100.0  # [N]

    # robot
    robot: RigidObjectCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot.spawn.rigid_props.disable_gravity = True
    # robot: ArticulationCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # legs: RigidObjectCfg = RigidObjectCfg(
    #     prim_path = "/World/envs/env_.*/Robot/FR_LEG/Cylinder",
    #     spawn = sim_utils.CylinderCfg(
    #         radius = 0.3,
    #         height = 0.05,
    #         activate_contact_sensors = True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props = sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic = 0.8),
    #         ),
    #         init_state = RigidObjectCfg.InitialStateCfg(),
    # )
    # camera
    # camera: CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/MainBody/Camera",
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -2.03/2), rot=rot_utils.euler_angles_to_quats(np.array([-90, 90, 0]), degrees=True).tolist(), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=64,
    #     height=64,
    # )
    # write_image_to_file = True

    ui_window_class_type = LanderEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity = (0.0, 0.0, -1.62),  # [m/s^2]
        physx=PhysxCfg(
            min_position_iteration_count=4,
            min_velocity_iteration_count=2,
            enable_stabilization=True, 
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"/workspace/isaaclab/source/lander_assets/moon_terrain_new.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=10, replicate_physics=True)

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/MainBody",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -1.23)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.1, 0.1]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/MainBody", 
        update_period=0.0, 
        track_air_time = True,
        debug_vis=True,
        history_length=5,
        filter_prim_paths_expr=["/World/ground"]
    )

    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/MainBody",
        update_period=0.0,
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=True,
        gravity_bias=(0, 0, 0),
    )

    # spaces
    action_space = 17 # 6 # 3D translational Fx,Fy,Fz,Mx,My,Mz
    state_space = 14
    observation_space = state_space # q0, q1, q2, q3, pos x, pos y, pos z, vel x, vel y, vel z, om_x, om_y, om_z, contact bool,

    # reward scales
    lin_vel_reward_scale = -1.3
    pos_reward_scale = -1.3 #-0.013
    du_reward_scale = -0.05
    mpower_reward_scale = -0.006 # -0.03
    rcspower_reward_scale = -0.003 #-0.03
    tpower_reward_scale = -0.3
    contact_reward_scale = 1.0
    du_reward_scale = -0.1

    vlim = 0.3  # [m/s] linear velocity limit for landing
    rlim = 2
    tlim = 2
    olim = 0.05
    prev_shaping = None


    # change viewer settings
    viewer = ViewerCfg(
        eye=(20.0, 20.0, 30.0),
        origin_type = "asset_body",
        asset_name = "robot",
        body_name = "MainBody",
        )

class LanderEnv(DirectRLEnv):
    cfg: LanderEnvCfg

    def __init__(self, cfg: LanderEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actionHigh = np.full(self.action_space.shape, 400, dtype=np.float32) # max thrust of RCS thrusters [N] and moment [Nm]
        self.actionLow = np.full(self.action_space.shape, 0, dtype=np.float32) # min thrust of RCS thrusters [N] and moment [Nm]
        self.actionLow[:,0] = 0.0
        self.actionHigh[:,0] = 43000.0 #43000.0
        self.action_space = gym.spaces.Box(dtype=np.float32, shape=self.actionHigh.shape ,low=self.actionLow, high=self.actionHigh)
        self.prev_action = torch.zeros(self.action_space.shape, device=self.device)
        self.d_action = torch.zeros(self.action_space.shape, device=self.device)
        self.aligned_history = torch.zeros((self.num_envs, 10), dtype=torch.bool, device=self.device)
        self._alignment_prev = torch.zeros(self.num_envs, device=self.device)
        self.landed_hist = 0
        self.crashed_hist = 0
        self.missed_hist = 0
        self.aligned_hist = 0
        self.hovering_hist = 0
        state_space = list(self.state_space.shape)
        state_space[1] -= 1               # modify element
        self._initial_state = torch.zeros(tuple(state_space), device=self.device)

        # d = 1.96 # distance from CG to each RCS thruster in meters
        # self.thruster_positions = torch.tensor([     # all positions in meters in body frame and from CG of vehicle
        #     [0.0, 0.0, 0.0],
        #     [d*torch.sin(torch.pi/4), d*torch.cos(torch.pi/4), 0.0],
        #     [d*torch.sin(torch.pi/4), d*torch.cos(torch.pi/4), 0.0],
        #     [d*torch.sin(torch.pi/4), d*torch.cos(torch.pi/4), 0.0],
        #     [d*torch.sin(torch.pi/4), d*torch.cos(torch.pi/4), 0.0],
        #     [d*torch.sin(3*torch.pi/4), d*torch.cos(3*torch.pi/4), 0.0],
        #     [d*torch.sin(3*torch.pi/4), d*torch.cos(3*torch.pi/4), 0.0],
        #     [d*torch.sin(3*torch.pi/4), d*torch.cos(3*torch.pi/4), 0.0],
        #     [d*torch.sin(3*torch.pi/4), d*torch.cos(3*torch.pi/4), 0.0],
        #     [d*torch.sin(5*torch.pi/4), d*torch.cos(5*torch.pi/4), 0.0],
        #     [d*torch.sin(5*torch.pi/4), d*torch.cos(5*torch.pi/4), 0.0],
        #     [d*torch.sin(5*torch.pi/4), d*torch.cos(5*torch.pi/4), 0.0],
        #     [d*torch.sin(5*torch.pi/4), d*torch.cos(5*torch.pi/4), 0.0],
        #     [d*torch.sin(7*torch.pi/4), d*torch.cos(7*torch.pi/4), 0.0],
        #     [d*torch.sin(7*torch.pi/4), d*torch.cos(7*torch.pi/4), 0.0],
        #     [d*torch.sin(7*torch.pi/4), d*torch.cos(7*torch.pi/4), 0.0],
        #     [d*torch.sin(7*torch.pi/4), d*torch.cos(7*torch.pi/4), 0.0],
        # ], device=self.device)

        d = 1.96

        angles = torch.tensor([
            0.0,
            3*torch.pi/4, 3*torch.pi/4, 3*torch.pi/4, 3*torch.pi/4,
            torch.pi/4, torch.pi/4, torch.pi/4, torch.pi/4,
            7*torch.pi/4, 7*torch.pi/4, 7*torch.pi/4, 7*torch.pi/4,   
            5*torch.pi/4, 5*torch.pi/4, 5*torch.pi/4, 5*torch.pi/4,
        ], device=self.device)

        x = d * torch.sin(angles)
        y = d * torch.cos(angles)
        z = torch.zeros_like(x)

        self.thruster_positions = torch.stack([x, y, z], dim=1)
        self.thruster_positions[0] = torch.tensor([0.0, 0.0, -1.4], device=self.device)  # main engine offset

        self.thruster_directions = torch.tensor([    # all directions in body frame (FRU)
            [0.0, 0.0, 1.0],  # main engine
            [0.0, 0.0, -1.0],  # Front Right - RCS engines start at top and rotate clockwise looking from CG of vehicle out
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],  # Front Left
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],  # Back Left
            [0.0, -1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],  # Back Right
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], device=self.device)

        # Total thrust and moment applied to the CoG of the lander
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device) # 3D
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device) # 3D

        # Logging
        self.episode_init = torch.zeros_like(self.episode_length_buf)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["reward"]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("MainBody")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        # idx = torch.tensor([0, 4, 8])
        # self._robot.data.default_inertia[:,idx] *= 3
        self._robot_inertia = self._robot.data.default_inertia[0].view(3,3)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot)
        # self._legs = RigidObject(self.cfg.legs)
        # self._robot = Articulation(self.cfg.robot)
        # self.scene.articulations["robot"] = self._robot
        # self._camera = Camera(self.cfg.camera)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._imu = Imu(self.cfg.imu)
        self.scene.rigid_objects["robot"] = self._robot
        # self.scene.rigid_objects["legs"] = self._legs
        # self.scene.sensors["camera"] = self._camera
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.scene.sensors["contact_forces"] = self._contact_sensor
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, 
                                           color=(0.75, 0.75, 0.75),
                                           texture_file ="/workspace/isaaclab/source/lander_assets/HDR_white_local_star.hdr",
                                           texture_format = "latlong",)
        light_cfg.func("/World/Light", light_cfg)
        dlight_cfg = sim_utils.DistantLightCfg(intensity=1000.0)
        dlight_cfg.func("/World/DistantLight", dlight_cfg)

    # takes normalised action and convert to real thrust and moment. Fz maps [-1,1] to [0, 1]
    def _pre_physics_step(self, actions: torch.Tensor):
        # self._actions = actions.clone().clamp(-1.0, 1.0)
        self._actions = actions.clone().clamp(torch.tensor(self.action_space.low, device=self.device), torch.tensor(self.action_space.high, device=self.device))
        # self.d_action = self._actions - self.prev_action
        # self.prev_action = self._actions.clone()
        # xthrust = self._actions[:,0]  # x thrust
        # ythrust = self._actions[:,1]  # y thrust
        # zthrust = self._actions[:,2]  # z thrust
        # xmoment = self._actions[:,3]  # x moment
        # ymoment = self._actions[:,4]  # y moment
        # zmoment = self._actions[:,5]  # z moment
        # thrusts = torch.stack([xthrust, ythrust, zthrust], dim=-1)  # [N, 3]
        # moments = torch.stack([xmoment, ymoment, zmoment], dim=-1)  # [N, 3]

        exclude = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], device=self.device)

        mask = torch.ones(17, dtype=torch.bool, device=self.device)
        mask[exclude] = False

        actions_filtered = self._actions[:, mask]                # (n, remaining)
        thruster_dirs_filtered = self.thruster_directions[mask]  # (remaining, 3)

        forces = actions_filtered @ thruster_dirs_filtered       # (n,3)

        # forces = self._actions @ self.thruster_directions
        forces_indv = self._actions.unsqueeze(-1) * self.thruster_directions.unsqueeze(0) # force from each thruster in body frame
        moments = torch.sum(torch.cross(self.thruster_positions.unsqueeze(0), forces_indv, axis=2),axis=1) # moment in body frame
        self._thrust[:, 0, :] = forces  # [N]
        self._moment[:, 0, :] = moments # [Nm]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        
        ray_hits_w = self._height_scanner.data.ray_hits_w  # shape (num_envs, 121, 3)

        # Extract z component at desired index
        z_values = ray_hits_w.mean(dim=1)  # shape (num_envs,)
        if torch.any(torch.isinf(z_values)):
            print("Warning: Inf values detected in raycast hits. Replacing with zeros.")
            z_values = torch.where(torch.isinf(z_values), torch.zeros_like(z_values), z_values)

        self._altitude = self._height_scanner.data.pos_w[..., -1] - z_values[:,-1] - 1.23  # for the convex hull
        self._quat = self._robot.data.root_quat_w
        self._pos = self._robot.data.root_pos_w
        self._pos[:,2] = self._altitude
        self._lin_vel = math.quat_apply(self._quat, self._imu.data.lin_vel_b)
        self._ang_vel = self._imu.data.ang_vel_b
                

        # lin_acc = self._imu.data.lin_acc_b
        # ang_acc = self._imu.data.ang_acc_b
        
        self._contact = self._contact_sensor.data.current_contact_time.squeeze(1)

        # print(f"camera_data: {camera_data.shape}")
        obs = torch.cat(
            [
                # camera_data.view(4, -1),       # [n, 30000]
                self._quat.view(self.num_envs, -1),      # [n, 4]
                self._pos.view(self.num_envs, -1),       # [n, 3]
                # lin_acc.view(self.num_envs, -1),           # [n, 3]
                # ang_acc.view(4, -1),           # [n, 3]
                self._lin_vel.view(self.num_envs, -1),           # [n, 3]
                self._ang_vel.view(self.num_envs, -1),           # [n, 3]
                # ang_vel.view(4, -1),           # [n, 3]
                self._contact.view(self.num_envs, -1),           # [4, 3]  ← squeeze out the 2nd dim
            ],
            dim=1
        )

        reward = self._get_rewards()

        ended, time_out = self._get_dones()
        # elementwise OR: if either landed or time_out is True
        is_last = time_out   # torch.Size([4])
        is_terminal = ended
        is_first = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
        dones = {"is_first": is_first, "is_last": is_last, "is_terminal": is_terminal}     
        
        observations = {"state": obs, "reward": reward}
        observations.update(dones)      
        return observations
    
    def _get_rewards(self) -> torch.Tensor:

        contact = self._contact_sensor.data.current_contact_time.squeeze(1)    
        reward = torch.zeros(self.num_envs, device=self.device)

        # --- Attitude Reward ---
        q_des = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        q_conj = torch.cat([self._quat[:, 0:1], -self._quat[:, 1:]], dim=1)
        e_q0 = q_conj[:, 0:1] * q_des[:, 0:1] - torch.sum(q_conj[:, 1:] * q_des[:, 1:], dim=1, keepdim=True)
        e_qv = q_conj[:, 0:1] * q_des[:, 1:] + q_des[:, 0:1] * q_conj[:, 1:] + torch.cross(q_conj[:, 1:], q_des[:, 1:], dim=1)
        self.alignment = 2.0 * torch.atan2(torch.norm(e_qv, dim=1), torch.abs(e_q0.squeeze(1)))
        self.alignment = torch.clamp(self.alignment, 0.0, torch.pi)
        self._aligned = self.alignment < 5.7e-2 #4.5e-4
        self.omega = torch.norm(self._ang_vel, dim=1)
        angle_delta = torch.abs(self.alignment - self._alignment_prev)
        self._alignment_prev = self.alignment.clone()

        # --- Alignment conditions ---
        self.aligned_history = torch.roll(self.aligned_history, shifts=-1, dims=1)
        self.aligned_history[:, -1] = self._aligned

        norm_actions = torch.norm(self._actions[:,1:], dim=1)/torch.tensor(self.actionHigh[:,1],device=self.device) # moment RCS penalty
        # du = torch.norm(self.d_action/self.actionHigh, dim=1)

        # --- Attitude reward ---
        alignment_penalty = (1/10)-1/(10*torch.exp(-self.alignment/(0.4)))
        rcs_penalty = -0.1*norm_actions #-0.03*norm_actions # - 0.3*du 
        ang_vel_penalty = -0.05 * (self._ang_vel.abs().sum(dim=1))

        reward = alignment_penalty.clone()
        reward += rcs_penalty
        reward += ang_vel_penalty


        # --- Translational reward ---
        
        # --- Hovering conditions ---
        alt_ok = self._altitude <= 2.0
        vel_ok = torch.norm(self._lin_vel, dim=1) < self.cfg.vlim
        pos_ok = torch.norm(self._pos[:, :2], dim=1) < self.cfg.rlim
        no_contact = contact <= 0.1
        self._hovering = alt_ok & vel_ok & pos_ok & no_contact

        # --- Landed conditions ---
        vel_landed = torch.abs(self._lin_vel[:, 2]) < self.cfg.vlim
        contact_landed = vel_ok & (contact > 0.5)
        self._landed = pos_ok & contact_landed

        # bounce = self._contact_sensor.data.current_air_time.squeeze(1)
        hard_landing = (contact > 0) & (~vel_landed)
        self._crashed = hard_landing

        # --- Missed conditions ---
        # self._missed = contact_landed & (~pos_ok) 
        # pos_for_reward = self._pos.clone()
        # pos_for_reward[:,2] = pos_for_reward[:,2]/self._initial_state[:,2]
        pos_reward = self.cfg.pos_reward_scale * torch.norm(self._pos, dim=1) #**2
        vel_reward = self.cfg.lin_vel_reward_scale * torch.norm(self._lin_vel, dim=1) #**2
        shaping = pos_reward + vel_reward # - 0 * np.linalg.norm(self._current_action - self._prev_action)**2

        if self.cfg.prev_shaping is not None:
            shaping_term = shaping - self.cfg.prev_shaping
            reward += shaping_term
        else:
            shaping_term = torch.zeros_like(reward)
        self.cfg.prev_shaping = shaping
        # reward += shaping
        

        main_engine_pen = -0.001*(self._actions[:,0]/torch.tensor(self.actionHigh[:,0],device=self.device))
        # rcs_translation_pen = -0.001*torch.norm(self._actions[:,0:2], dim=1)/torch.tensor(self.actionHigh[:,0],device=self.device)
        reward += main_engine_pen #+ rcs_translation_pen

        self.out_of_bounds_x = torch.logical_or(self._robot.data.root_pos_w[:,0] > 40, self._robot.data.root_pos_w[:,0] < -40)
        self.out_of_bounds_y = torch.logical_or(self._robot.data.root_pos_w[:,1] > 40, self._robot.data.root_pos_w[:,1] < -40)
        self.out_of_bounds = torch.logical_or(self.out_of_bounds_x, self.out_of_bounds_y)

        # --- Penalties and Bonuses ---
        reward[angle_delta > np.deg2rad(30)] -= 50 # penalise action more instead
        # reward[~self._aligned & (self.omega > np.deg2rad(0.5)) & (self._landed | self._crashed)] = -500
        # reward[~self._aligned & self._missed] = 0
        # reward[~self._aligned & self._landed] = -30
        reward[self._landed] += 200
        reward[~self._aligned & self._crashed] -= 100 # 40
        # reward[self.out_of_bounds] -= 40
        # reward -= 0.1
        # reward[~self._aligned & self._crashed] = -50 # reward[~self._aligned & (self._crashed | self._missed)] = -40
        # reward[self._aligned] += 0.3
        hovering_pen = 0.001*self._actions[pos_ok,0]
        reward[pos_ok] -= hovering_pen
        reward[self.aligned_history.all(dim=1) & self._landed] += 500

        for i in range(self.num_envs):
            roll, pitch, yaw = math.euler_xyz_from_quat(self._quat)
            if self._hovering[i]:
                reward[i] -= 0.1*torch.norm(self._actions[i,0])
                print(f"Env {i} Hovering")
            if self._landed[i]:
                print(f"""Env {i} Landed with:
                    Position [m]             {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
                    Velocity [m/s]           {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
                    Euler Angles [deg]       {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
                    Alignment [deg]          {self.alignment[i]:.4f}
                    Angular Velocity [rad/s] {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
                    Contact Time             {contact[i]*self.step_dt:.2f}s
                    at time                  {self.episode_length_buf[i] * self.step_dt:.2f}s""")
            elif self._crashed[i]:
                print(f"""Env {i} Crashed with:
                    Position [m]             {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
                    Velocity [m/s]           {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
                    Euler Angles [deg]       {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
                    Alignment [deg]          {self.alignment[i]:.4f}
                    Angular Velocity [rad/s] {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
                    Contact Time             {contact[i]*self.step_dt:.2f}s
                    at time                  {self.episode_length_buf[i] * self.step_dt:.2f}s""")
        

        rewards = {"reward": reward}

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        if self.num_envs == 8:
            with torch.no_grad():

                # Bonus/Penalty events
                self.landed_hist += (self._landed).sum().item()
                self.aligned_hist += (self._aligned).sum().item()
                self.crashed_hist += (self._crashed).sum().item()

                # Summary statistics (mean/std)
                print(f"\n=== Reward Diagnostics ===")
                print(f"Attitude term:       mean={alignment_penalty.mean():.3f}, std={alignment_penalty.std():.3f}")
                print(f"RCS term:            mean={rcs_penalty.mean():.3f}, std={rcs_penalty.std():.3f}")
                print(f"Angular Vel term:    mean={ang_vel_penalty.mean():.3f}, std={ang_vel_penalty.std():.3f}")
                print(f"Position term:       mean={pos_reward.mean():.3f}, std={pos_reward.std():.3f}")
                print(f"Velocity term:       mean={vel_reward.mean():.3f}, std={vel_reward.std():.3f}")
                # print(f"Main Engine term:    mean={main_engine_pen.mean():.3f}, std={main_engine_pen.std():.3f}")
                print(f"--- Event counts ---")
                print(f"Landed: {self.landed_hist}, Aligned: {self.aligned_hist}, Crashed: {self.crashed_hist}")
                print(f"Total reward mean:  {reward.mean():.3f}, std={reward.std():.3f}")
                print("==========================\n")



        return reward
    
    # def _get_rewards(self) -> torch.Tensor:

    #     contact = self._contact_sensor.data.current_contact_time.squeeze(1)
    
    #     reward = torch.zeros(self.num_envs, device=self.device)

    #     # --- Attitude Reward ---
    #     q_des = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
    #     q_conj = torch.cat([self._quat[:, 0:1], -self._quat[:, 1:]], dim=1)
    #     e_q0 = q_conj[:, 0:1] * q_des[:, 0:1] - torch.sum(q_conj[:, 1:] * q_des[:, 1:], dim=1, keepdim=True)
    #     e_qv = q_conj[:, 0:1] * q_des[:, 1:] + q_des[:, 0:1] * q_conj[:, 1:] + torch.cross(q_conj[:, 1:], q_des[:, 1:], dim=1)
    #     self.alignment = 2.0 * torch.atan2(torch.norm(e_qv, dim=1), torch.abs(e_q0.squeeze(1)))
    #     self.alignment = self.alignment**2
    #     self._aligned = self.alignment < (5.7e-2)**2 #4.5e-4
    #     self.omega = torch.norm(self._ang_vel, dim=1)
    #     angle_delta = torch.abs(self.alignment - self._alignment_prev)
    #     self._alignment_prev = self.alignment.clone()

    #     # --- Alignment conditions ---
    #     self.aligned_history = torch.roll(self.aligned_history, shifts=-1, dims=1)
    #     self.aligned_history[:, -1] = self._aligned

    #     norm_actions = torch.norm(self._actions/torch.tensor(self.actionHigh, device=self.device), dim=1)

    #     # --- Attitude reward ---
    #     error_pos = torch.norm(self._pos/self._initial_state[:,0:3].abs(), dim=1)**2
    #     root_position_error = 0.15*torch.exp(-0.1*(error_pos+0.5*self.alignment))
    #     error_vel = torch.norm(self._lin_vel/self._initial_state[:,7:10].abs(), dim=1)**2
    #     error_ang_vel = torch.norm(self._ang_vel/self._initial_state[:,10:13].abs(), dim=1)**2
    #     root_velocity_error = 0.1*torch.exp(-0.1*(error_vel+0.1*error_ang_vel))
    #     rcs_penalty = -0.3*norm_actions # - 0.3*du

    #     reward += root_position_error + root_velocity_error + rcs_penalty 

    #     # --- Translational reward ---
        
    #     # --- Hovering conditions ---
    #     alt_ok = self._altitude <= 2.0
    #     vel_ok = torch.norm(self._lin_vel, dim=1) < self.cfg.vlim
    #     pos_ok = torch.norm(self._pos[:, :2], dim=1) < self.cfg.rlim
    #     no_contact = contact <= 0.1
    #     # self._hovering = alt_ok & vel_ok & pos_ok & no_contact

    #     # --- Landed conditions ---
    #     vel_landed = torch.abs(self._lin_vel[:, 2]) < self.cfg.vlim
    #     contact_landed = vel_ok & (contact > 0.5)
    #     self._landed = pos_ok & contact_landed

    #     hard_landing = (contact > 0) & (~vel_landed)
    #     self._crashed = hard_landing


    #     self.out_of_bounds_x = torch.logical_or(self._robot.data.root_pos_w[:,0] > 40, self._robot.data.root_pos_w[:,0] < -40)
    #     self.out_of_bounds_y = torch.logical_or(self._robot.data.root_pos_w[:,1] > 40, self._robot.data.root_pos_w[:,1] < -40)
    #     self.out_of_bounds = torch.logical_or(self.out_of_bounds_x, self.out_of_bounds_y)

    #     # --- Penalties and Bonuses ---
    #     # reward[angle_delta > np.deg2rad(30)] -= 50
    #     # reward[~self._aligned & (self.omega > np.deg2rad(0.5)) & (self._landed | self._crashed)] = -500
    #     # reward[~self._aligned & self._missed] = 0
    #     reward[~self._aligned & self._landed] -= 10
    #     reward[self._landed & self._aligned] += 50
    #     reward[self._crashed] -= 100
    #     reward[self.out_of_bounds] -= 100
    #     reward[self._aligned & self._landed] += 200
    #     # reward[~self._aligned & self._crashed] = -200 # reward[~self._aligned & (self._crashed | self._missed)] = -40
    #     # reward[pos_ok] -= 1*self._actions[pos_ok,0]
    #     # reward[~alt_ok] -= 0.1
    #     reward[self.aligned_history.all(dim=1) & self._landed] += 500

    #     for i in range(self.num_envs):
    #         roll, pitch, yaw = math.euler_xyz_from_quat(self._quat)
    #         # if self._hovering[i]:
    #         #     reward[i] -= 0.05*torch.norm(self._actions[i,:3])
    #         #     print(f"Env {i} Hovering")
    #         if self._landed[i]:
    #             print(f"""Env {i} Landed with:
    #                 Position [m]             {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
    #                 Velocity [m/s]           {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
    #                 Euler Angles [deg]       {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
    #                 Alignment [deg]          {self.alignment[i]:.4f}
    #                 Angular Velocity [rad/s] {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
    #                 Contact Time             {contact[i]*self.step_dt:.2f}s
    #                 at time                  {self.episode_length_buf[i] * self.step_dt:.2f}s""")
    #         elif self._crashed[i]:
    #             print(f"""Env {i} Crashed with:
    #                 Position [m]             {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
    #                 Velocity [m/s]           {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
    #                 Euler Angles [deg]       {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
    #                 Alignment [deg]          {self.alignment[i]:.4f}
    #                 Angular Velocity [rad/s] {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
    #                 Contact Time             {contact[i]*self.step_dt:.2f}s
    #                 at time                  {self.episode_length_buf[i] * self.step_dt:.2f}s""")
    #         # elif self._missed[i]:
    #         #     print(f"""Env {i} Missed with:
    #         #         Position [m]             {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
    #         #         Velocity [m/s]           {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
    #         #         Euler Angles [deg]       {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
    #         #         Alignment [deg]          {self.alignment[i]:.4f}
    #         #         Angular Velocity [rad/s] {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
    #         #         Contact Time             {contact[i]*self.step_dt:.2f}s
    #         #         at time                  {self.episode_length_buf[i] * self.step_dt:.2f}s""")

    #     rewards = {"reward": reward}

    #     # Logging
    #     for key, value in rewards.items():
    #         self._episode_sums[key] += value

    #     with torch.no_grad():

    #         # Bonus/Penalty events
    #         self.landed_hist += (self._landed).sum().item()
    #         self.aligned_hist += (self._aligned).sum().item()
    #         self.crashed_hist += (self._crashed).sum().item()

    #         # Summary statistics (mean/std)
    #         print(f"\n=== Reward Diagnostics ===")
    #         # print(f"Attitude term:       mean={alignment_penalty.mean():.3f}, std={alignment_penalty.std():.3f}")
    #         # print(f"RCS term:            mean={rcs_penalty.mean():.3f}, std={rcs_penalty.std():.3f}")
    #         # print(f"Angular Vel term:    mean={ang_vel_penalty.mean():.3f}, std={ang_vel_penalty.std():.3f}")
    #         # print(f"Position term:       mean={pos_reward.mean():.3f}, std={pos_reward.std():.3f}")
    #         # print(f"Velocity term:       mean={vel_reward.mean():.3f}, std={vel_reward.std():.3f}")
    #         # print(f"Shaping term:        mean={shaping_term.mean():.3f}, std={shaping_term.std():.3f}")
    #         # print(f"Main Engine term:    mean={main_engine_pen.mean():.3f}, std={main_engine_pen.std():.3f}")
    #         print(f"Root Position Error:  mean={root_position_error.mean():.3f}, std={root_position_error.std():.3f}")
    #         print(f"Root Velocity Error:  mean={root_velocity_error.mean():.3f}, std={root_velocity_error.std():.3f}")
    #         print(f"Action Penalty:      mean={rcs_penalty.mean():.3f}, std={rcs_penalty.std():.3f}     ")
    #         print(f"--- Event counts ---")
    #         print(f"Landed: {self.landed_hist}, Aligned: {self.aligned_hist}, Crashed: {self.crashed_hist}")
    #         print(f"Total reward mean:  {reward.mean():.3f}, std={reward.std():.3f}")
    #         print("==========================\n")

    #     return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.time_out = (self.episode_length_buf >= self.max_episode_length - 1)

        # self.terminated = torch.logical_or(self._crashed, self._missed)
        self.terminated = torch.logical_or(self._crashed, self._landed)
        self.terminated = torch.logical_or(self.terminated, self.out_of_bounds)

        return self.terminated, self.time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self.aligned_history[env_ids,:] = False
        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)


        self._robot.reset(env_ids) # necessary for isaaclab
        self._imu.reset(env_ids) # necessary for isaaclab
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            
        # self.episode_init[env_ids] = self.episode_length_buf[env_ids]
        # if len(env_ids) == self.num_envs:
        #     self.episode_init[2] -= 10
        
        
        # self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(0.0, 0.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0, 0)
        # Reset robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        # joint_vel = self._robot.data.default_joint_vel[env_ids]
        init_euler = torch.zeros(len(env_ids), 3, device=self.device).uniform_(-5*np.pi/180, 5*np.pi/180) # roll, pitch, yawv +- 5 degrees
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :2] += torch.zeros_like(default_root_state[:, :2]).uniform_(-10,10)#(-20.0, 20.0) # x and y position
        default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(60,80)#(0.0, 20.0) # z position
        default_root_state[:, 3:7] = math.quat_from_euler_xyz(init_euler[:,0], init_euler[:,1], init_euler[:,2])  # random orientation
        default_root_state[:, 7:9] += torch.zeros_like(default_root_state[:, 7:9]).uniform_(-1.0, 1.0) # x and y linear velocity
        default_root_state[:, 9] += torch.zeros_like(default_root_state[:, 9]).uniform_(-3.0, -1.0) # z linear velocity
        default_root_state[:, 10:13] += torch.zeros_like(default_root_state[:, 10:13]).uniform_(-0.017, 0.017) # angular velocity
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._initial_state[env_ids] = default_root_state
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
