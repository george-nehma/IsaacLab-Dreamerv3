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
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
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
    decimation = 2
    episode_length_s = 30.0
    debug_vis = True
    # action_scale = 100.0  # [N]

    # robot
    robot: RigidObjectCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # camera
    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/MainBody/Camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -2.03/2), rot=rot_utils.euler_angles_to_quats(np.array([-90, 90, 0]), degrees=True).tolist(), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=64,
        height=64,
    )
    write_image_to_file = True

    ui_window_class_type = LanderEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity = (0.0, 0.0, -1.62),  # [m/s^2]
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
        usd_path=f"/workspace/isaaclab/source/lander_assets/moon_terrain_smooth.usd",
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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
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
    action_space = 5 # 3D translational
    state_space = 6
    observation_space = (camera.height*camera.width*3) + 1 + 3 + 3 + 3  # rgb, height, lin_acc, ang_acc, lin_vel, ang_vel, contact

    # reward scales
    lin_vel_reward_scale = -100
    altitude_reward_scale = -100
    mpower_reward_scale = -0.6
    spower_reward_scale = -0.3
    # ang_vel_reward_scale = -0.01
    vlim = 0.5  # [m/s] linear velocity limit for landing

    # change viewer settings
    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales

class LanderEnv(DirectRLEnv):
    cfg: LanderEnvCfg

    def __init__(self, cfg: LanderEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the CoG of the lander
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device) # 3D
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device) # 3D

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "altitude",
                "lin_vel",
                "power",
            ]
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
        self._camera = Camera(self.cfg.camera)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self._contact_forces = ContactSensor(self.cfg.contact_forces)
        self._imu = Imu(self.cfg.imu)
        self.scene.rigid_objects["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.scene.sensors["contact_forces"] = self._contact_forces
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=10.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # takes normalised action and convert to real thrust and moment. Fz maps [-1,1] to [0, 1]
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        xthrust = self._actions[:,0] - self._actions[:,1]  # x thrust
        ythrust = self._actions[:,2] - self._actions[:,3]
        zthrust = self._actions[:,4]  # z thrust
        thrusts = torch.stack([xthrust, ythrust, zthrust], dim=-1)  # [N, 3]
        self._thrust[:, 0, :] = 0.5 * (thrusts + 1) * (3000 - 0) + 0  # Fx maps [-1,1] to [0, 3000] [N]
        # self._moment[:, 0, :] = 0 * self.cfg.moment_scale * self._actions[:, 1:] # don't update moment in 2D env but pass through as zero

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self, is_first) -> dict:
        data_type = "rgb"
        if "rgb" in self.cfg.camera.data_types:
            camera_data = self._camera.data.output[data_type] / 255.0
            if self.cfg.write_image_to_file:
                save_images_to_file(camera_data, f"lander_cam_{data_type}.png")
            # normalize the camera data for better training results
            mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            camera_data -= mean_tensor
        
        height_data = (self._height_scanner.data.pos_w[..., -1] - self._height_scanner.data.ray_hits_w[:, 60, -1] - 2.03/2).unsqueeze(-1)
        lin_acc = self._imu.data.lin_acc_b
        ang_acc = self._imu.data.ang_acc_b
        lin_vel = self._imu.data.lin_vel_b
        ang_vel = self._imu.data.ang_vel_b
        contact = self._contact_forces.data.net_forces_w.squeeze(1)

        # print(f"camera_data: {camera_data.shape}")
        obs = torch.cat(
            [
                # camera_data.view(4, -1),       # [4, 30000]
                height_data.view(self.num_envs, -1),       # [4, 1]
                lin_acc.view(self.num_envs, -1),           # [4, 3]
                # ang_acc.view(4, -1),           # [4, 3]
                lin_vel.view(self.num_envs, -1),           # [4, 3]
                # ang_vel.view(4, -1),           # [4, 3]
                contact.view(self.num_envs, -1),           # [4, 3]  ← squeeze out the 2nd dim
            ],
            dim=1
        )

        landed, time_out = self._get_dones()
        # elementwise OR: if either landed or time_out is True
        is_last = landed | time_out   # torch.Size([4])
        is_terminal = landed | time_out
            
        dones = {"is_first": is_first, "is_last": is_last, "is_terminal": is_terminal}

        # if not torch.isfinite(obs).all():
        #     is_last = True
        #     is_terminal = True
        # print(f"Height : {height_data}\nAction: {self._actions}")
        reward = self._get_rewards()
        
        observations = {"image": camera_data, "state": obs, "reward": reward}
        observations.update(dones)      
        return observations

    def _get_rewards(self) -> torch.Tensor:

        lin_acc = self._imu.data.lin_acc_b
        ang_acc = self._imu.data.ang_acc_b
        lin_vel = torch.linalg.norm(self._imu.data.lin_vel_b, dim=1)
        ang_vel = self._imu.data.ang_vel_b 
        contact_forces = self._contact_forces.data.net_forces_w_history.squeeze(2)  # Extract the z-component of the contact forces
        contact = torch.all(contact_forces[...,2] > 0)
        altitude = self._height_scanner.data.pos_w[..., -1] - self._height_scanner.data.ray_hits_w[:, 60, -1] - 2.03/2

        self._mpower = (self._actions[:,4] != 0).to(dtype=torch.int, device=self.device)
        self._spower = (self._actions[:, :4] != 0).any(dim=1).to(dtype=torch.int, device=self.device)
    

        landed = (torch.all(altitude <= 0) and contact and
            torch.all(lin_vel < self.cfg.vlim))  

        crashed = (torch.all(altitude <= 0) and contact and
            torch.all(lin_vel > self.cfg.vlim)) 
            
        missed = (torch.all(altitude <= 0) and contact and
            torch.all(lin_vel < self.cfg.vlim))

        if landed:
            bonus = torch.tensor([100], device=self.device)
            print("Landed!")
        elif crashed:
            bonus = torch.tensor([-100], device=self.device)
            print("Crashed!")
        elif missed:
            bonus = torch.tensor([0], device=self.device)
            print("Missed!")
        else:
            bonus = torch.tensor([0], device=self.device)

        
        rewards = {
            "altitude": altitude * self.cfg.altitude_reward_scale,
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale + bonus,
            "power": self._mpower * self.cfg.mpower_reward_scale + self._spower * self.cfg.spower_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # # debug print 
        # for i, x in enumerate([rewards["lin_vel"], rewards["altitude"], rewards["power"]]):
        #     print(f"Item {i}: shape={getattr(x, 'shape', None)}, dtype={getattr(x, 'dtype', None)}, device={getattr(x, 'device', None)}")
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1
        lin_vel = self._imu.data.lin_vel_b
        contact_forces = self._contact_forces.data.net_forces_w_history.squeeze(2)  # Extract the z-component of the contact forces
        contact = torch.all(contact_forces[...,2] > 0, dim=1)
        altitude = self._height_scanner.data.pos_w[..., -1] - self._height_scanner.data.ray_hits_w[:, 60, -1] - 2.03/2

        vel_ok = torch.linalg.norm(lin_vel, dim = 1) < self.cfg.vlim
        self.landed = torch.logical_and(altitude < 2 , contact == True)
        self.landed = altitude <= 2
        # print(f"Altitude: {altitude}, Contact: {contact}, Vel: {vel_ok}")
        # crashed  = altitude <= 0 & contact & torch.all(torch.linalg.norm(lin_vel) > self.cfg.vlim)
        return self.landed, self.time_out

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

        self._robot.reset(env_ids)
        self._imu.reset(env_ids)
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
        default_root_state[:, :2] += torch.zeros_like(default_root_state[:, :2]).uniform_(-10.0, 10.0) # x and y position
        default_root_state[:, 2] += torch.zeros_like(default_root_state[:, 2]).uniform_(0.0, 20.0) # z position
        default_root_state[:, 7:9] += torch.zeros_like(default_root_state[:, 7:9]).uniform_(-10.0, 10.0) # x and y linear velocity
        default_root_state[:, 9] += torch.zeros_like(default_root_state[:, 9]).uniform_(-10.0, 10.0) # z linear velocity
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
