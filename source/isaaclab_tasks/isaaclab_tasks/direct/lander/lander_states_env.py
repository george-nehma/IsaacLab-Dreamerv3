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
from isaaclab.utils import math
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


class LanderStatesEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: LanderStatesEnv, window_name: str = "IsaacLab"):
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
class LanderStatesEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 12
    episode_length_s = 90.0
    debug_vis = True
    # action_scale = 100.0  # [N]

    # robot
    robot: RigidObjectCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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

    ui_window_class_type = LanderStatesEnvWindow

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=25, replicate_physics=True)

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/MainBody",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -2.03/2)),
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
    state_space = 7
    observation_space = state_space # pos x, pos y, pos z, vel x, vel y, vel z, contact bool,

    # reward scales
    lin_vel_reward_scale = -1
    pos_reward_scale = -1
    mpower_reward_scale = -0.006
    spower_reward_scale = -0.003
    contact_reward_scale = 1.0
    du_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    vlim = 0.3  # [m/s] linear velocity limit for landing
    rlim = 2
    prev_shaping = None


    # change viewer settings
    viewer = ViewerCfg(
        eye=(20.0, 20.0, 30.0),
        origin_type = "asset_body",
        asset_name = "robot",
        body_name = "MainBody",
        )

class LanderStatesEnv(DirectRLEnv):
    cfg: LanderStatesEnvCfg

    def __init__(self, cfg: LanderStatesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actionHigh = np.full(self.action_space.shape, 800, dtype=np.float32) # max thrust of RCS thrusters [N] and moment 
        self.actionLow = np.full(self.action_space.shape, -800, dtype=np.float32) # min thrust of RCS thrusters [N] and moment
        self.actionLow[:,-1] = 0.0
        self.actionHigh[:,-1] = 43000.0
        self.action_space = gym.spaces.Box(dtype=np.float32, shape=self.actionHigh.shape ,low=self.actionLow, high=self.actionHigh)
        self.prev_action = torch.zeros(self.action_space.shape, device=self.device)

        # Total thrust and moment applied to the CoG of the lander
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device) # 3D
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device) # 3D

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["reward"]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("MainBody")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot)
        # self._robot = Articulation(self.cfg.robot)
        # self.scene.articulations["robot"] = self._robot
        # self._camera = Camera(self.cfg.camera)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._imu = Imu(self.cfg.imu)
        self.scene.rigid_objects["robot"] = self._robot
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
        # xthrust = self._actions[:,0] - self._actions[:,1]  # x thrust
        # ythrust = self._actions[:,2] - self._actions[:,3]
        # zthrust = self._actions[:,4]  # z thrust
        xthrust = self._actions[:,0]  # x thrust
        ythrust = self._actions[:,1]  # y thrust
        zthrust = self._actions[:,2]  # z thrust
        thrusts = torch.stack([xthrust, ythrust, zthrust], dim=-1)  # [N, 3]
        self._thrust[:, 0, :] = thrusts  # [N]
        # self._moment[:, 0, :] = 0 * self.cfg.moment_scale * self._actions[:, 1:] # don't update moment in 2D env but pass through as zero

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        
        ray_hits_w = self._height_scanner.data.ray_hits_w  # shape (num_envs, 121, 3)

        # Extract z component at desired index
        z_values = ray_hits_w.mean(dim=1)  # shape (num_envs,)
        
        self._quat = self._robot.data.root_quat_w
        self._altitude = self._height_scanner.data.pos_w[..., -1] - z_values[:,-1] - 2.03/2  # for the convex hull
        self._pos = self._robot.data.root_pos_w
        self._pos[:,2] = self._altitude
        self._lin_vel = math.quat_apply(self._quat, self._imu.data.lin_vel_b)
                
        if torch.isinf(self._altitude).any():
            print("altitude is -inf")
        # lin_acc = self._imu.data.lin_acc_b
        # ang_acc = self._imu.data.ang_acc_b
        
        # ang_vel = self._imu.data.ang_vel_b
        self._contact = self._contact_sensor.data.current_contact_time.squeeze(1)

        # print(f"camera_data: {camera_data.shape}")
        obs = torch.cat(
            [
                # camera_data.view(4, -1),       # [n, 30000]
                self._pos.view(self.num_envs, -1),       # [n, 3]
                # lin_acc.view(self.num_envs, -1),           # [n, 3]
                # ang_acc.view(4, -1),           # [n, 3]
                self._lin_vel.view(self.num_envs, -1),           # [n, 3]
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
        
        self._mpower = (self._actions[:,2] != 0).to(dtype=torch.int, device=self.device) # changed from 4 to 2
        self._spower = (self._actions[:, :2] != 0).any(dim=1).to(dtype=torch.int, device=self.device)
    
        reward = torch.zeros(self.num_envs, device=self.device)

        # --- Hovering conditions ---
        alt_ok = self._altitude <= 2.0
        vel_ok = torch.norm(self._lin_vel, dim=1) < self.cfg.vlim
        pos_ok = torch.norm(self._pos[:, :2], dim=1) < self.cfg.rlim
        no_contact = contact <= 0.1
        self._hovering = alt_ok & vel_ok & pos_ok & no_contact

        # --- Landed conditions ---
        vel_landed = torch.abs(self._lin_vel[:, 2]) < self.cfg.vlim
        contact_landed = vel_landed & (contact > 0.5)
        self._landed = pos_ok & contact_landed

        # --- Crashed conditions ---
        tilt_thresh = (1.0 - torch.cos(torch.deg2rad(torch.tensor(15.0, device=self.device)))) / 2.0
        xy_sq = self._robot.data.root_quat_w[..., 1]**2 + self._robot.data.root_quat_w[..., 2]**2
        tilted = xy_sq > tilt_thresh

        hard_landing = (contact > 0) & (~vel_landed)
        self._crashed = hard_landing | tilted

        # --- Missed conditions ---
        self._missed = contact_landed & (~pos_ok) 

        pos_reward = self.cfg.pos_reward_scale * torch.norm(self._pos, dim=1)
        vel_reward = self.cfg.lin_vel_reward_scale * torch.norm(self._lin_vel, dim=1)
        shaping = pos_reward + vel_reward # - 0 * np.linalg.norm(self._current_action - self._prev_action)**2

        if self.cfg.prev_shaping is not None:
            reward = shaping - self.cfg.prev_shaping
        self.cfg.prev_shaping = shaping

        reward += self.cfg.spower_reward_scale * self._spower + self.cfg.mpower_reward_scale * self._mpower 

        reward += 1 * torch.where(self._lin_vel[:,2] > 0, -torch.ones_like(self._lin_vel[:,2]), torch.zeros_like(self._lin_vel[:,2]))

        mask_contact = (~self._crashed) & (contact > 0.5)
        reward[mask_contact] += self.cfg.contact_reward_scale * contact[mask_contact]

        # du = torch.norm(self._actions - self.prev_action, dim=1)
        # reward += self.cfg.du_reward_scale * du
        # self.prev_action = self._actions.clone()

        reward[self._landed] = 30
        reward[self._crashed] = -40
        reward[self._missed] = 0

        for i in range(self.num_envs):
            # if mask_contact[i]:
            #     print(f"Env {i} Contact with Position {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}, Velocity {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f} at time {self.episode_length_buf[i]*self.sim.cfg.dt:.2f}s")
            if self._hovering[i]:
                reward[i] -= 0.01*torch.norm(self._actions[i,:])
            #     print(f"Env {i} Hovering with Position {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}, Velocity {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f} at time {self.episode_length_buf[i]*self.sim.cfg.dt:.2f}s")
            if self._landed[i]:
                print(f"""Env {i} Landed with:
                    Position [m]             {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
                    Velocity [m/s]           {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
                    Contact Time             {contact[i]*self.step_dt:.2f}s
                    at time                  {self.episode_length_buf[i] * self.step_dt:.2f}s""")
            elif self._crashed[i]:
                print(f"""Env {i} Crashed with:
                    Position [m]      {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
                    Velocity [m/s]    {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
                    Contact Time      {contact[i]*self.step_dt:.2f}s
                    at time           {self.episode_length_buf[i] * self.step_dt:.2f}s""")
            elif self._missed[i]:
                print(f"""Env {i} Missed with:
                    Position [m]      {self._pos[i][0]:.2f}, {self._pos[i][1]:.2f}, {self._pos[i][2]:.2f}
                    Velocity [m/s]    {self._lin_vel[i][0]:.2f}, {self._lin_vel[i][1]:.2f}, {self._lin_vel[i][2]:.2f}
                    at time           {self.episode_length_buf[i] * self.step_dt:.2f}s""")
                
        rewards = {"reward": reward}

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.out_of_bounds_x = torch.logical_or(self._robot.data.root_pos_w[:,0] > 40, self._robot.data.root_pos_w[:,0] < -40)
        self.out_of_bounds_y = torch.logical_or(self._robot.data.root_pos_w[:,1] > 40, self._robot.data.root_pos_w[:,1] < -40)
        self.out_of_bounds = torch.logical_or(self.out_of_bounds_x, self.out_of_bounds_y)
    
        self.terminated = torch.logical_or(self._crashed, self._missed)
        self.terminated = torch.logical_or(self.terminated, self._landed)
        self.terminated = torch.logical_or(self.terminated, self.out_of_bounds)

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
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(0.0, 0.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0, 0)
        # Reset robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        # joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :2] += torch.zeros_like(default_root_state[:, :2]).uniform_(-20,20)#(-20.0, 20.0) # x and y position
        default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(60,80)#(0.0, 20.0) # z position
        default_root_state[:, 7:9] += torch.zeros_like(default_root_state[:, 7:9]).uniform_(-5.0, 5.0) # x and y linear velocity
        default_root_state[:, 9] += torch.zeros_like(default_root_state[:, 9]).uniform_(-30.0, -20.0) # z linear velocity
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
