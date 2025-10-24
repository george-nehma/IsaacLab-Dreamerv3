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


class Lander6DOFEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: Lander6DOFEnv, window_name: str = "IsaacLab"):
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
class Lander6DOFEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 12
    episode_length_s = 90.0
    debug_vis = True
    # action_scale = 100.0  # [N]

    # robot
    robot: RigidObjectCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
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

    ui_window_class_type = Lander6DOFEnvWindow

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
    action_space = 3 # 3D translational
    state_space = 8
    observation_space = state_space # q0, q1, q2, q3, pos x, pos y, pos z, vel x, vel y, vel z, om_x, om_y, om_z, contact bool,

    # reward scales
    lin_vel_reward_scale = -100
    pos_reward_scale = -100
    ang_vel_reward_scale = -10
    ang_reward_scale = -200
    mpower_reward_scale = -0.6
    spower_reward_scale = -0.3
    tpower_reward_scale = -0.1
    contact_reward_scale = 100.0
    du_reward_scale = -0.1

    # pos_reward_scale = 0.8
    # lin_vel_reward_scale = 0.5
    # ang_reward_scale = 0.8
    # ang_vel_reward_scale = 0.3

    # ctrl_cost_scale = 0.012      # control magnitude cost
    # spower_penalty_scale = 0.01
    # mpower_penalty_scale = 0.05
    # tpower_penalty_scale = 0.05
    # hover_ctrl_penalty = 0.2

    # up_vel_penalty = 1.0        # penalty for positive vz
    # contact_reward_scale = 1.0

    # attitude_tol_rad = 0.03     # ~8.6 degrees for landed
    # bad_align_rad = 0.5         # large penalty if > ~28.6 degrees near ground
    # alt_tol = 2.0               # altitude threshold for hover/land checks

    # success_reward = 50.0
    # crash_penalty = 20.0
    # bad_align_penalty = 40.0

    # reward_clip = 100.0







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

class Lander6DOFEnv(DirectRLEnv):
    cfg: Lander6DOFEnvCfg

    def __init__(self, cfg: Lander6DOFEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actionHigh = np.full(self.action_space.shape, 1000, dtype=np.float32) # max thrust of RCS thrusters [N] and moment [Nm]
        self.actionLow = np.full(self.action_space.shape, -1000, dtype=np.float32) # min thrust of RCS thrusters [N] and moment [Nm]
        # self.actionLow[:,:2] = -0.0 # [Nm]
        # self.actionHigh[:,:2] = 0.0 # [Nm]
        # self.actionLow[:,2] = 0.0
        # self.actionHigh[:,2] = 0.0
        self.action_space = gym.spaces.Box(dtype=np.float32, shape=self.actionHigh.shape ,low=self.actionLow, high=self.actionHigh)
        self.prev_action = torch.zeros(self.action_space.shape, device=self.device)
        self.d_action = torch.zeros(self.action_space.shape, device=self.device)
        self._quat_prev = torch.zeros(self.num_envs, 4, device=self.device)

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
        self.d_action = self._actions - self.prev_action
        self.prev_action = self._actions.clone()
        # xthrust = self._actions[:,0] - self._actions[:,1]  # x thrust
        # ythrust = self._actions[:,2] - self._actions[:,3]
        # zthrust = self._actions[:,4]  # z thrust
        # xthrust = self._actions[:,0]  # x thrust
        # ythrust = self._actions[:,1]  # y thrust
        # zthrust = self._actions[:,2]  # z thrust
        xmoment = self._actions[:,0]  # x moment
        ymoment = self._actions[:,1]  # y moment
        zmoment = self._actions[:,2]  # z moment
        # thrusts = torch.stack([xthrust, ythrust, zthrust], dim=-1)  # [N, 3]
        moments = torch.stack([xmoment, ymoment, zmoment], dim=-1)  # [N, 3]
        # self._thrust[:, 0, :] = thrusts  # [N]
        self._moment[:, 0, :] = moments # don't update moment in 2D env but pass through as zero

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        
        ray_hits_w = self._height_scanner.data.ray_hits_w  # shape (num_envs, 121, 3)
        # num_envs = ray_hits_w.shape[0]
        # idx = 60

        # Extract z component at desired index
        z_values = ray_hits_w.mean(dim=1)  # shape (num_envs,)

        # Find which ones are inf
        # mask_inf = torch.isinf(z_values)

        # if mask_inf.any():
        #     idx_inf = torch.nonzero(mask_inf, as_tuple=True)[0]
        #     print(f"broken array {ray_hits_w[idx_inf,:,:]}")# For each env where z is inf, find closest non-inf along the ray
        #     non_inf_mask = ~torch.isinf(ray_hits_w[:, :, -1])  # shape (num_envs, 121)
        #     non_inf_indices = torch.arange(ray_hits_w.shape[1], device=ray_hits_w.device)  # 0..120

        #     for i in torch.nonzero(mask_inf).squeeze():
        #         valid_indices = non_inf_indices[non_inf_mask[i]]
        #         closest_idx = valid_indices[(valid_indices - idx).abs().argmin()]
        #         z_values[i] = ray_hits_w[i, closest_idx, -1]

        self._altitude = self._height_scanner.data.pos_w[..., -1] - z_values[:,-1] - 1.23  # for the convex hull
        self._quat = self._robot.data.root_quat_w
        self._pos = self._robot.data.root_pos_w
        self._pos[:,2] = self._altitude
        self._lin_vel = math.quat_apply(self._quat, self._imu.data.lin_vel_b)
        self._ang_vel = self._imu.data.ang_vel_b
                
        if torch.isinf(self._altitude).any():
            print("altitude is -inf")
        # lin_acc = self._imu.data.lin_acc_b
        # ang_acc = self._imu.data.ang_acc_b
        
        self._contact = self._contact_sensor.data.current_contact_time.squeeze(1)

        # print(f"camera_data: {camera_data.shape}")
        obs = torch.cat(
            [
                # camera_data.view(4, -1),       # [n, 30000]
                self._quat.view(self.num_envs, -1),      # [n, 4]
                # self._pos.view(self.num_envs, -1),       # [n, 3]
                # lin_acc.view(self.num_envs, -1),           # [n, 3]
                # ang_acc.view(4, -1),           # [n, 3]
                # self._lin_vel.view(self.num_envs, -1),           # [n, 3]
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
        
        # self._mpower = (self._actions[:,2] != 0).to(dtype=torch.int, device=self.device) # changed from 4 to 2
        # self._spower = (self._actions[:, :2] != 0).any(dim=1).to(dtype=torch.int, device=self.device)
        self._tpower = (self._actions[:, :] != 0).any(dim=1).to(dtype=torch.int, device=self.device)
    
        reward = torch.zeros(self.num_envs, device=self.device)

        roll,pitch,yaw = math.euler_xyz_from_quat(self._quat)  # shape (3)
        e_roll = 2*(1-torch.cos(math.wrap_to_pi(roll)))
        e_pitch = 2*(1-torch.cos(math.wrap_to_pi(pitch)))
        e_yaw = 2*(1-torch.cos(math.wrap_to_pi(yaw)))
        e_angle = torch.stack([e_roll, e_pitch, e_yaw], dim=1)
        alignment = torch.sum(e_angle, dim=1)
        # v_unit = self._lin_vel / (torch.norm(self._lin_vel, dim=1, keepdim=True) + 1e-8)
        # z_axis_body = torch.tensor([0, 0, -1], dtype=torch.float, device=self.device).expand(self.num_envs, -1)
        # z_world = math.quat_apply(self._quat, z_axis_body)
        # alignment = torch.sum(v_unit * z_world, dim=1)

        # q_des = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device).expand(self.num_envs, -1)
        # self_conj = torch.cat([self._quat[:, 0:1], -self._quat[:, 1:]], dim=1)
        # e_q0 = self_conj[:,0:1]*q_des[:,0:1] - torch.sum(self_conj[:,1:]*q_des[:,1:], dim=1, keepdim=True)
        # e_qv = self_conj[:,0:1]*q_des[:,1:] + q_des[:,0:1]*self_conj[:,1:] + torch.cross(self_conj[:,1:], q_des[:,1:], dim=1)
        # d = torch.sum(self._quat * q_des, dim=-1)                 # shape (...)
        # d = torch.abs(d)                                # account for q ~ -q
        # d = torch.clamp(d, -1.0, 1.0)
        # alignment = 2.0 * torch.atan2(torch.norm(e_qv, dim=1), e_q0.squeeze(1))
        # print(f"Alignment: {alignment[0]:.4f}")

        # tilt_thresh = (1.0 - torch.cos(torch.deg2rad(torch.tensor(2.0, device=self.device)))) / 2.0
        # xy_sq = self._robot.data.root_quat_w[..., 1]**2 + self._robot.data.root_quat_w[..., 2]**2
        # algn_ok = xy_sq < tilt_thresh

        # --- Hovering conditions ---
        algn_ok = alignment < 4.5e-4
        # algn_ok = alignment < 3.01e-2 # within 1 degree on each axis
        # algn_ok = torch.norm(torch.stack([roll, pitch], dim=1), dim=1) < torch.deg2rad(1.0)
        # alt_ok = self._altitude <= 2.0
        # vel_ok = torch.norm(self._lin_vel, dim=1) < self.cfg.vlim
        # pos_ok = torch.norm(self._pos[:, :2], dim=1) < self.cfg.rlim
        # no_contact = contact <= 0.1
        # self._hovering = algn_ok & alt_ok & vel_ok & pos_ok & no_contact

        # --- Landed conditions ---
        # vel_landed = torch.abs(self._lin_vel[:, 2]) < self.cfg.vlim
        # contact_landed = vel_landed & (contact > 0.5)
        # self._landed = algn_ok & pos_ok & contact_landed

        # --- Crashed conditions ---
        # hard_landing = (contact > 0) & (~vel_landed)
        # tilted_landing = (~algn_ok) & alt_ok
        # self._crashed = hard_landing | tilted_landing

        self._bad_align = ~algn_ok #& (self._altitude <5.0)
        self._aligned = algn_ok # & (self._altitude <5.0)
        self._too_slow = (self.episode_length_buf > 100 + self.episode_init) & self._bad_align 


        # --- Missed conditions ---
        # self._missed = algn_ok & contact_landed & (~pos_ok) 

        # shaping = (self.cfg.pos_reward_scale * torch.norm(self._pos, dim=1) 
                    # + self.cfg.lin_vel_reward_scale * torch.norm(self._lin_vel, dim=1)

        

        ang_reward = self.cfg.ang_reward_scale * alignment/(torch.pi)
        ang_vel_reward = self.cfg.ang_vel_reward_scale * torch.norm(self._ang_vel, dim=1)
        # reward = ang_reward + ang_vel_reward

        # if self.cfg.prev_shaping is not None:
        #     reward = shaping - self.cfg.prev_shaping
        # self.cfg.prev_shaping = shaping

        # reward += (self.cfg.spower_reward_scale * self._spower + 
        #             self.cfg.mpower_reward_scale * self._mpower + 
        #             self.cfg.tpower_reward_scale * self._tpower)
        norm_actions = torch.norm(self._actions, dim=1)/self.actionHigh.max()
        # reward += (self.cfg.tpower_reward_scale * norm_actions)
        du = torch.norm(self.d_action, dim=1)/self.actionHigh.max()
        # reward += self.cfg.du_reward_scale * du

        # reward = 2*torch.exp(-alignment/(0.04*2*torch.pi)) - 0.3*norm_actions - 0.3*du
        reward = -alignment - 0.3*norm_actions - 0.3*du - 0.01*torch.norm(self._ang_vel, dim=1)
        reward[(self._quat[:,0] > self._quat_prev[:,0])] -= 1
        # print(f"change in angle: {self._quat[:,0] - self._quat_prev[:,0][0]:.4f}")
        self._quat_prev = self._quat.clone()
        # reward += 100 * torch.where(self._lin_vel[:,2] > 0, -torch.ones_like(self._lin_vel[:,2]), torch.zeros_like(self._lin_vel[:,2])) # penalty for positive z velocity

        # mask_contact = (~self._crashed) & (contact > 0.5)
        # reward[mask_contact] += self.cfg.contact_reward_scale * contact[mask_contact]

        # for i in range(self.num_envs):
        #     # if mask_contact[i]:
        #     #     print(f"Env {i} Contact with Position {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}, Velocity {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f} at time {self.episode_length_buf[i]*self.sim.cfg.dt:.2f}s")
        #     # if self._hovering[i]:
        #         # reward[i] -= 1*torch.norm(self._actions[i,:])
        #     #     print(f"Env {i} Hovering with Position {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}, Velocity {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f} at time {self.episode_length_buf[i]*self.sim.cfg.dt:.2f}s")
        #     if self._landed[i]:
        #         print(f"""Env {i} Landed with:
        #             Position          {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
        #             Velocity          {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
        #             Euler Angles      {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
        #             Angular Velocity  {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
        #             at time           {self.episode_length_buf[i] * self.step_dt:.2f}s""")
        #     elif self._crashed[i]:
        #         print(f"""Env {i} Crashed with:
        #             Position          {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
        #             Velocity          {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
        #             Euler Angles      {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
        #             Angular Velocity  {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
        #             at time           {self.episode_length_buf[i] * self.step_dt:.2f}s""")
        #     elif self._missed[i]:
        #         print(f"""Env {i} Missed with:
        #             Position          {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
        #             Velocity          {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
        #             Euler Angles      {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
        #             Angular Velocity  {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
        #             at time           {self.episode_length_buf[i] * self.step_dt:.2f}s""")

        # reward[self._landed] = 3000
        # reward[self._crashed] = -4000
        # reward[self._missed] += 0
        reward[~algn_ok & (torch.norm(self._ang_vel) > 0.5)  & (self.episode_length_buf > 100 + self.episode_init)] = -25
        reward[self._too_slow] = 0
        reward[self._aligned] += 9
        reward[self._aligned & (self.episode_length_buf > 100 + self.episode_init)] = 50
        # reward += 3000.0 * torch.exp(-2000 * torch.abs(alignment-8.98e-2)) 

        rewards = {"reward": reward}

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward



    # def _get_rewards(self) -> torch.Tensor:
    #     device = self.device
    #     eps = 1e-8

    #     contact = self._contact_sensor.data.current_contact_time.squeeze(1)  # (N,)

    #     # boolean flags for power usage (0/1)
    #     mpower = (self._actions[:, 2] != 0).to(dtype=torch.float32, device=device)
    #     spower = (self._actions[:, :2] != 0).any(dim=1).to(dtype=torch.float32, device=device)
    #     tpower = (self._actions[:, 3:] != 0).any(dim=1).to(dtype=torch.float32, device=device)

    #     N = self.num_envs
    #     reward = torch.zeros(N, device=device)

    #     # Euler for logging only (keep as-is)
    #     roll, pitch, yaw = math.euler_xyz_from_quat(self._quat)

    #     # Desired quaternion (no rotation)
    #     q_des = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).expand(N, -1)
    #     dot = torch.sum(self._quat * q_des, dim=-1)
    #     dot = torch.clamp(torch.abs(dot), -1.0 + eps, 1.0 - eps)
    #     alignment = 2.0 * torch.acos(dot)   # angle in radians, shape (N,)

    #     # --- logical condition masks (tunable thresholds in cfg) ---
    #     # Use sensible thresholds (radians)
    #     algn_ok = alignment < self.cfg.attitude_tol_rad     # e.g., 0.15 rad (~8.6deg)
    #     alt_ok = self._altitude <= self.cfg.alt_tol         # e.g., 2.0 m
    #     vel_ok = torch.norm(self._lin_vel, dim=1) < self.cfg.vlim
    #     pos_ok = torch.norm(self._pos[:, :2], dim=1) < self.cfg.rlim
    #     no_contact = contact <= 0.1

    #     hovering = algn_ok & alt_ok & vel_ok & pos_ok & no_contact

    #     # landed / crashed / missed
    #     vel_landed = torch.abs(self._lin_vel[:, 2]) < self.cfg.vlim
    #     contact_landed = (vel_landed) & (contact > 0.5)
    #     self._landed = algn_ok & pos_ok & contact_landed

    #     hard_landing = (contact > 0) & (~vel_landed)
    #     tilted_landing = (~algn_ok) & alt_ok
    #     self._crashed = hard_landing | tilted_landing

    #     self._bad_align = (alignment > self.cfg.bad_align_rad) & (self._altitude < 5.0)
    #     self._missed = algn_ok & contact_landed & (~pos_ok)

    #     # --- shaping potential (positive = worse). We will do reward = prev_shaping - shaping ---
    #     pos_norm = torch.norm(self._pos, dim=1)
    #     lin_vel_norm = torch.norm(self._lin_vel, dim=1)
    #     ang_vel_norm = torch.norm(self._ang_vel, dim=1)

    #     shaping = (
    #         self.cfg.pos_reward_scale * pos_norm
    #         + self.cfg.lin_vel_reward_scale * lin_vel_norm
    #         + self.cfg.ang_reward_scale * (alignment ** 2)
    #         + self.cfg.ang_vel_reward_scale * ang_vel_norm
    #     )

    #     # initialize prev_shaping if None (per-env)
    #     if getattr(self.cfg, "prev_shaping", None) is None:
    #         self.cfg.prev_shaping = torch.zeros_like(shaping)

    #     # reward from potential difference: positive when shaping decreases (improvement)
    #     reward += (self.cfg.prev_shaping - shaping)
    #     self.cfg.prev_shaping = shaping.detach()

    #     # --- control / power penalties (negative) ---
    #     # control magnitude penalty (vectorized)
    #     action_mag = torch.norm(self._actions, dim=1)  # or use per-component normalized sum if preferred
    #     reward -= self.cfg.ctrl_cost_scale * action_mag

    #     # small penalty for turning on thrusters (binary), treat as penalties
    #     reward -= (self.cfg.spower_penalty_scale * spower
    #             + self.cfg.mpower_penalty_scale * mpower
    #             + self.cfg.tpower_penalty_scale * tpower)

    #     # penalty for upward vertical velocity (if you want)
    #     # we only penalize upward velocity when above target altitude or to discourage upward after approaching pad
    #     vz = self._lin_vel[:, 2]
    #     reward += torch.where(vz > 0, -self.cfg.up_vel_penalty, torch.zeros_like(vz))

    #     # contact shaping: mild positive for gentle contact if not crashed
    #     mask_contact = (~self._crashed) & (contact > 0.5)
    #     reward[mask_contact] += self.cfg.contact_reward_scale * contact[mask_contact]

    #     # hovering control penalty (encourage less thrust when hovering)
    #     if hovering.any():
    #         idx = hovering.nonzero(as_tuple=False).squeeze(-1)
    #         reward[idx] -= self.cfg.hover_ctrl_penalty * torch.norm(self._actions[idx, :], dim=1)

    #     # Terminal sparse rewards (moderate magnitudes)
    #     # Add once (not overwrite) so shaping still matters
    #     R_success = self.cfg.success_reward  # e.g., +50.0
    #     R_crash = -abs(self.cfg.crash_penalty)  # e.g., -20.0
    #     R_bad_align = -abs(self.cfg.bad_align_penalty)  # e.g., -40.0

    #     reward = reward.clone()
    #     reward[self._landed] += R_success
    #     reward[self._crashed] += R_crash
    #     reward[self._missed] += 0.0
    #     reward[self._bad_align] += R_bad_align

    #     # Clip reward to reasonable range (helps Dreamer)
    #     reward = torch.clamp(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

    #     # Logging sums
    #     rewards = {"reward": reward}

    #     # Logging
    #     for key, value in rewards.items():
    #         self._episode_sums[key] += value
    #     return reward






    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.out_of_bounds_x = torch.logical_or(self._robot.data.root_pos_w[:,0] > 40, self._robot.data.root_pos_w[:,0] < -40)
        self.out_of_bounds_y = torch.logical_or(self._robot.data.root_pos_w[:,1] > 40, self._robot.data.root_pos_w[:,1] < -40)
        self.out_of_bounds = torch.logical_or(self.out_of_bounds_x, self.out_of_bounds_y)
    
        # self.terminated = torch.logical_or(self._crashed, self._missed)
        # self.terminated = torch.logical_or(self.terminated, self._bad_align)
        # self.terminated = torch.logical_or(self.terminated, self._aligned)
        # self.terminated = torch.logical_or(self.terminated, self._landed)
        self.terminated = torch.logical_or(self._too_slow, self.out_of_bounds)

        # if self.terminated.any():
        #     idx = self.terminated.nonzero(as_tuple=False).squeeze(-1)
        #     roll, pitch, yaw = math.euler_xyz_from_quat(self._quat)
        #     for i in idx:
        #         print(f"""Env {i} Terminated with:
        #             Position          {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
        #             Velocity          {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
        #             Euler Angles      {torch.rad2deg(roll[i]):.2f}, {torch.rad2deg(pitch[i]):.2f}, {torch.rad2deg(yaw[i]):.2f}
        #             Angular Velocity  {self._ang_vel[i][0]:.2f}, {self._ang_vel[i][1]:.2f}, {self._ang_vel[i][2]:.2f}
        #             at time           {self.episode_length_buf[i] * self.step_dt:.2f}s""")


        return self.terminated, self.time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES


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
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length)-100)
            
        self.episode_init[env_ids] = self.episode_length_buf[env_ids]
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
        init_euler = torch.zeros(len(env_ids), 3, device=self.device).uniform_(-10*np.pi/180, 10*np.pi/180) # roll, pitch, yawv +- 5 degrees
        # init_euler[:,2] = 0
        # init_euler[:,1] = 0
        # init_quat = math.quat_from_euler_xyz(init_euler[:,0], init_euler[:,1], init_euler[:,2])  
        default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :2] += torch.zeros_like(default_root_state[:, :2]).uniform_(-20,20)#(-20.0, 20.0) # x and y position
        default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(60,80)#(0.0, 20.0) # z position
        default_root_state[:, 3:7] = math.quat_from_euler_xyz(init_euler[:,0], init_euler[:,1], init_euler[:,2])  # random orientation
        # default_root_state[:, 7:9] += torch.zeros_like(default_root_state[:, 7:9]).uniform_(-5.0, 5.0) # x and y linear velocity
        # default_root_state[:, 9] += torch.zeros_like(default_root_state[:, 9]).uniform_(-30.0, -20.0) # z linear velocity
        default_root_state[:, 10:13] += torch.zeros_like(default_root_state[:, 10:13]).uniform_(-0.035, 0.035) # angular velocity
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
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
