# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="This simulation allows for teleoperated control of a 6DOF lander using a joystick.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import pathlib

import pygame

from packaging import version

import numpy as np
import ruamel.yaml as yaml

isaaclab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, isaaclab_root)

from source.DreamerRL import exploration as expl
from source.DreamerRL import models
from source.DreamerRL import tools
from source.DreamerRL.envs import wrappers
from source.DreamerRL.parallel import Damy

import torch
from torch import nn
from torch import distributions as torchd

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pickle


from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict

from source.DreamerRL.isaaclab_wrapper import IsaacLabDreamerWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow, ViewportCameraController
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math
from isaaclab.sensors import Camera, CameraCfg, save_images_to_file, ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, Imu, ImuCfg, patterns
import isaacsim.core.utils.numpy.rotations as rot_utils


from source.lander_assets.lander_vehicle_rgd import LUNAR_LANDER_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# PLACEHOLDER: Extension template (do not remove this comment)


to_np = lambda x: x.detach().cpu().numpy()


@configclass
class TeleopLanderCfg(DirectRLEnvCfg):
    """Configuration for a cart-pole scene."""

    robot: RigidObjectCfg = LUNAR_LANDER_CFG.replace(prim_path="/World/Robot")

    # sim: SimulationCfg = SimulationCfg(
    #     dt=1 / 120,
    #     render_interval=1,
    #     gravity = (0.0, 0.0, -1.62),  # [m/s^2] 3.73
    #     physx=PhysxCfg(
    #         min_position_iteration_count=4,
    #         min_velocity_iteration_count=2,
    #         enable_stabilization=True, 
    #     ),
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    # )
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

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/Robot/MainBody",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -1.23)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.1, 0.1]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/Robot/MainBody", 
        update_period=0.0, 
        track_air_time = True,
        debug_vis=True,
        history_length=5,
        filter_prim_paths_expr=["/World/ground"]
    )

    imu: ImuCfg = ImuCfg(
        prim_path="/World/Robot/MainBody",
        update_period=0.0,
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=True,
        gravity_bias=(0, 0, 0),
    )

    viewer = ViewerCfg(
        eye=(-20.0, 0.0, 15.0),
        origin_type = "asset_body",
        asset_name = "robot",
        body_name = "MainBody",
        )

class Lander6DOFEnv(DirectRLEnv):
    cfg: TeleopLanderCfg

    def __init__(self, cfg: TeleopLanderCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actionHigh = np.full(self.action_space.shape, 1000, dtype=np.float32) # max thrust of RCS thrusters [N] and moment [Nm]
        self.actionLow = np.full(self.action_space.shape, -1000, dtype=np.float32) # min thrust of RCS thrusters [N] and moment [Nm]
        self.actionLow[:,0] = 0.0
        self.actionHigh[:,0] = 43000.0
        self.action_space = gym.spaces.Box(dtype=np.float32, shape=self.actionHigh.shape ,low=self.actionLow, high=self.actionHigh)
        self.prev_action = torch.zeros(self.action_space.shape, device=self.device)
        self.d_action = torch.zeros(self.action_space.shape, device=self.device)
        self.aligned_history = torch.zeros((self.num_envs, 10), dtype=torch.bool, device=self.device)
        self._alignment_prev = torch.zeros(self.num_envs, device=self.device)
  

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


def get_observation(robot, imu, height_scanner, contact_sensor, marker_visualizer, action, time):
        
    ray_hits_w = height_scanner.data.ray_hits_w  # shape (num_envs, 121, 3)

    # Extract z component at desired index
    z_values = ray_hits_w.mean(dim=1)  # shape (num_envs,)

    altitude = height_scanner.data.pos_w[..., -1] - z_values[:,-1] - 1.23  # for the convex hull
    quat = robot.data.root_quat_w
    pos = robot.data.root_pos_w
    pos[:,2] = altitude
    lin_vel = math.quat_apply(quat, imu.data.lin_vel_b)
    ang_vel = imu.data.ang_vel_b

    # lin_acc = imu.data.lin_acc_b
    # ang_acc = imu.data.ang_acc_b
    
    contact = contact_sensor.data.current_contact_time.squeeze(1)

    obs = torch.cat(
        [
            # camera_data.view(4, -1),       # [n, 30000]
            quat.view(1, -1),      # [n, 4]
            pos.view(1, -1),       # [n, 3]
            # lin_acc.view(num_envs, -1),           # [n, 3]
            # ang_acc.view(4, -1),           # [n, 3]
            lin_vel.view(1, -1),           # [n, 3]
            ang_vel.view(1, -1),           # [n, 3]
            # ang_vel.view(4, -1),           # [n, 3]
            contact.view(1, -1),           # [4, 3]  ← squeeze out the 2nd dim
        ],
        dim=1
    )

    reward, state = get_rewards(obs, contact_sensor, action)

    ended, time_out = get_dones(robot,state, time)
    # elementwise OR: if either landed or time_out is True
    is_last = time_out   # torch.Size([4])
    is_terminal = ended
        
    dones = {"is_last": is_last, "is_terminal": is_terminal}     
    
    observations = {"state": obs, "reward": reward}
    observations.update(dones)    

    marker_visualizer.visualize(pos, quat, marker_indices=torch.tensor([0]))  
    return observations

def get_rewards(obs, contact_sensor, action):

    quat = obs[:, 0:4]
    pos = obs[:, 4:7]
    lin_vel = obs[:, 7:10]
    ang_vel = obs[:, 10:13]
    altitude = pos[:, 2]


    contact = contact_sensor.data.current_contact_time.squeeze(1)
    
    mpower = (action[:,0] != 0).to(dtype=torch.int) # changed from 4 to 2
    # spower = (actions[:, 1:] != 0).any(dim=1).to(dtype=torch.int, device=device)
    # tpower = (actions[:, :] != 0).any(dim=1).to(dtype=torch.int, device=device)

    reward = torch.zeros(1)

    # --- Attitude Reward ---
    q_des = torch.tensor([1.0, 0.0, 0.0, 0.0],device='cuda:0').expand(1, -1)
    q_conj = torch.cat([quat[:, 0:1], -quat[:, 1:]], dim=1)
    e_q0 = q_conj[:, 0:1] * q_des[:, 0:1] - torch.sum(q_conj[:, 1:] * q_des[:, 1:], dim=1, keepdim=True)
    e_qv = q_conj[:, 0:1] * q_des[:, 1:] + q_des[:, 0:1] * q_conj[:, 1:] + torch.cross(q_conj[:, 1:], q_des[:, 1:], dim=1)
    alignment = 2.0 * torch.atan2(torch.norm(e_qv, dim=1), torch.abs(e_q0.squeeze(1)))
    alignment = torch.clamp(alignment, 0.0, torch.pi)
    aligned = alignment < 5.7e-2 #4.5e-4

    # --- Alignment conditions ---

    norm_actions = torch.norm(action[:,1:], dim=1)/1000
    # du = torch.norm(d_action/actionHigh, dim=1)

    # --- Attitude reward ---
    reward = (1/10)-1/(10*torch.exp(-alignment/(0.4)))
    reward -= 0.03*norm_actions # - 0.3*du 
    reward -= 0.05*ang_vel[:,0].abs()
    reward -= 0.05*ang_vel[:,1].abs()
    reward -= 0.05*ang_vel[:,2].abs()


    # --- Translational reward ---
    
    # --- Hovering conditions ---
    alt_ok = altitude <= 2.0
    vel_ok = torch.norm(lin_vel, dim=1) < 0.3
    pos_ok = torch.norm(pos[:, :2], dim=1) < 2
    no_contact = contact <= 0.1
    # hovering = alt_ok & vel_ok & pos_ok & no_contact

    # --- Landed conditions ---
    vel_landed = torch.abs(lin_vel[:, 2]) < 0.3
    contact_landed = vel_ok & (contact > 0.5)
    landed = pos_ok & contact_landed

    hard_landing = (contact > 0) & (~vel_landed)
    crashed = hard_landing

    pos_reward = -0.07 * torch.norm(pos, dim=1)
    vel_reward = -0.13 * torch.norm(lin_vel, dim=1)
    shaping = pos_reward + vel_reward # - 0 * np.linalg.norm(current_action - prev_action)**2

    reward += shaping

    reward += -0.06 * mpower  #+ cfg.mpower_reward_scale * mpower 

    # --- Penalties and Bonuses ---
    reward[landed] += 4
    reward[crashed] -= 40


    roll, pitch, yaw = math.euler_xyz_from_quat(quat)
    if landed:
        print(f"""Env Landed with:
            Position [m]             {pos[0][0]:.2f}, {pos[0][1]:.2f}, {pos[0][2]:.2f}
            Velocity [m/s]           {lin_vel[0][0]:.2f}, {lin_vel[0][1]:.2f}, {lin_vel[0][2]:.2f}
            Euler Angles [deg]       {torch.rad2deg(roll[0]):.2f}, {torch.rad2deg(pitch[0]):.2f}, {torch.rad2deg(yaw[0]):.2f}
            Alignment [deg]          {alignment[0]:.4f}
            Angular Velocity [rad/s] {ang_vel[0][0]:.2f}, {ang_vel[0][1]:.2f}, {ang_vel[0][2]:.2f}""")
        
    elif crashed:
        print(f"""Env Crashed with:
            Position [m]             {pos[0][0]:.2f}, {pos[0][1]:.2f}, {pos[0][2]:.2f}
            Velocity [m/s]           {lin_vel[0][0]:.2f}, {lin_vel[0][1]:.2f}, {lin_vel[0][2]:.2f}
            Euler Angles [deg]       {torch.rad2deg(roll[0]):.2f}, {torch.rad2deg(pitch[0]):.2f}, {torch.rad2deg(yaw[0]):.2f}
            Alignment [deg]          {alignment[0]:.4f}
            Angular Velocity [rad/s] {ang_vel[0][0]:.2f}, {ang_vel[0][1]:.2f}, {ang_vel[0][2]:.2f}""")

    state = {"landed": landed, "crashed": crashed}
    return reward, state

def get_dones(robot,state, time) -> tuple[torch.Tensor, torch.Tensor]:

    time_out = (time >= 180 - 1)

    crashed = state["crashed"]
    landed = state["landed"]
    out_of_bounds_x = torch.logical_or(robot.data.root_pos_w[:,0] > 40, robot.data.root_pos_w[:,0] < -40)
    out_of_bounds_y = torch.logical_or(robot.data.root_pos_w[:,1] > 40, robot.data.root_pos_w[:,1] < -40)
    out_of_bounds = torch.logical_or(out_of_bounds_x, out_of_bounds_y)

    # terminated = torch.logical_or(crashed, missed)
    terminated = torch.logical_or(crashed, landed)
    terminated = torch.logical_or(terminated, out_of_bounds)

    return terminated, time_out


def unnormalize(x_norm, min_val, max_val):
        return 0.5 * (x_norm + 1) * (max_val - min_val) + min_val


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(5, 5, 5),
            ),
        }
    )
    marker = VisualizationMarkers(marker_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, 
                                           color=(0.75, 0.75, 0.75),
                                           texture_file ="/workspace/isaaclab/source/lander_assets/HDR_white_local_star.hdr",
                                           texture_format = "latlong",)
    light_cfg.func("/World/Light", light_cfg)
    dlight_cfg = sim_utils.DistantLightCfg(intensity=1000.0)
    dlight_cfg.func("/World/DistantLight", dlight_cfg)

    # # change viewer settings
    # viewer = ViewerCfg(
    #     eye=(-20.0, 0.0, 30.0),
    #     origin_type = "asset_body",
    #     asset_name = "robot",
    #     body_name = "MainBody",
    #     )

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick connected.")
        return
    joystick = pygame.joystick.Joystick(0)

    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        scene.update(dt=sim_dt)
        # Reset
        if count % 1000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            init_euler = torch.zeros(1, 3).uniform_(-10*np.pi/180, 10*np.pi/180) # roll, pitch, yawv +- 5 degrees

            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.terrain.env_origins
            root_state[:, :2] += torch.zeros_like(root_state[:, :2]).uniform_(-20,20)#(-20.0, 20.0) # x and y position
            root_state[:, 2] += torch.zeros_like(root_state[:, 2]).uniform_(60,80)#(0.0, 20.0) # z position
            root_state[:, 3:7] = math.quat_from_euler_xyz(init_euler[:,0], init_euler[:,1], init_euler[:,2])  # random orientation
            root_state[:, 7:9] += torch.zeros_like(root_state[:, 7:9]).uniform_(-5.0, 5.0) # x and y linear velocity
            root_state[:, 9] += torch.zeros_like(root_state[:, 9]).uniform_(-20.0, -10.0) # z linear velocity
            root_state[:, 10:13] += torch.zeros_like(root_state[:, 10:13]).uniform_(-0.035, 0.035) # angular velocity

            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        pygame.event.pump()  # Process event queue

        main_engine = unnormalize(joystick.get_axis(5),0,4000) # -1 to 1
        pitch = unnormalize(joystick.get_axis(1),-1000,1000)  # -1 to 1
        roll = unnormalize(joystick.get_axis(0),-1000,1000)   # -1 to 1
        yaw = unnormalize(joystick.get_axis(3),-1000,1000)    # -1 to 1

        print(f"""
              Main Engine: {main_engine:.2f}, 
              Pitch: {pitch:.2f}, 
              Roll: {roll:.2f}, 
              Yaw: {yaw:.2f}
            """)

        action = torch.tensor([[main_engine, pitch, roll, yaw]],device='cuda:0')

        time = count*sim_dt
        obs = get_observation(scene["robot"], scene["imu"], scene["height_scanner"], scene["contact_sensor"], marker, action, time)
        # Apply random action

        if obs["is_last"] | obs["is_terminal"]:
            break

        thrust = torch.zeros(1, 1, 3,device='cuda:0')
        moment = torch.zeros(1, 1, 3,device='cuda:0')

        zthrust = action[:,0]  # z thrust
        xmoment = action[:,1]  # x moment
        ymoment = action[:,2]  # y moment
        zmoment = action[:,3]  # z moment
        thrusts = torch.stack([0*zthrust, 0*zthrust, zthrust], dim=-1)  # [N, 3]
        moments = torch.stack([xmoment, ymoment, zmoment], dim=-1)  # [N, 3]
        thrust[:, 0, :] = thrusts  # [N]
        moment[:, 0, :] = moments # don't update moment in 2D env but pass through as zero

        # -- apply action to the robot
        body_id = scene["robot"].find_bodies("MainBody")[0]
        scene["robot"].set_external_force_and_torque(thrust, moment, body_ids=body_id)
        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(states, actions, rewards, timesteps, dt=0.01, save_prefix="traj"):
    """
    Plots state trajectories, control actions, rewards, and contact over time.

    Args:
        states (ndarray): (n,7) or (n,14). Order:
                          (7)  = [x, y, z, vx, vy, vz, contact]
                          (8)  = [qw, qx, qy, qz, wx, wy, wz, contact]
                          (14) = [qw, qx, qy, qz, x, y, z, vx, vy, vz, wx, wy, wz, contact]
        actions (ndarray): (n,3) or (n,6).
        rewards (ndarray): (n,).
        dt (float): timestep size.
        save_prefix (str): prefix for saved figures.
    """
    n, sdim = states.shape
    
    if sdim == 8 or sdim == 14:
        w = states[:, 0]
        x = states[:, 1]
        y = states[:, 2]
        z = states[:, 3]
        # Roll (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0) # Clamp t2 to avoid invalid arcsin input
        pitch = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        euler_angles = np.stack([roll*180/np.pi, pitch*180/np.pi, yaw*180/np.pi], axis=1)
        states = np.concatenate((euler_angles, states[:, 4:]), axis=1) if (states.shape[1] == 14 or states.shape[1] == 8) else states
        states[:,-3:] = states[:,-3:]*180/np.pi # convert angular velocity to deg/s

        timesteps = np.arange(states.shape[0]) * dt
        n, sdim = states.shape
        _, adim = actions.shape

    # --- State labels ---

    if sdim == 7:
        state_labels = ["x [m]", "y [m]", "z [m]", "vx [m/s]", "vy [m/s]", "vz [m/s]", "contact"]
        action_labels = ["Fx [N]", "Fy [N]", "Fz [N]"]
        plot_titles = ["Fx [N]", "Fy [N]", "Fz [N]"]
        _, adim = actions.shape
    
    elif sdim == 14:
        state_labels = ["qw", "qx", "qy", "qz",
                        "x [m]", "y [m]", "z [m]",
                        "vx [m/s]", "vy [m/s]", "vz [m/s]",
                        "wx [deg/s]", "wy [deg/s]", "wz [deg/s]",
                        "contact"]
        action_labels = [ "Fx [N]", "Fy [N]", "Fz [N]", "Mx [Nm]", "My [Nm]", "Mz [Nm]"]
        plot_titles = ["Fx [N]", "Fy [N]", "Fz [N]", "Fx + Mx [N/Nm]", " Fx + My [N/Nm]", "Mz [Nm]"]
        

    elif sdim == 13:
        state_labels = ["roll [deg]", "pitch [deg]", "yaw [deg]",
                        "x [m]", "y [m]", "z [m]",
                        "vx [m/s]", "vy [m/s]", "vz [m/s]",
                        "wx [deg/s]", "wy [deg/s]", "wz [deg/s]",
                        "contact"]
        action_labels = [ "Fx [N]", "Fy [N]", "Fz [N]", "Mx [Nm]", "My [Nm]", "Mz [Nm]"]
        plot_titles = ["Fx [N]", "Fy [N]", "Fz [N]", "Fx + Mx [N/Nm]", " Fx + My [N/Nm]", "Mz [Nm]"]
        
    else:
        raise ValueError("States must be (n,7), (n,8) or (n,14).")

    

    # --- Plot all states dynamically ---
    nrows = int(np.ceil(sdim / 2))
    fig1, axes1 = plt.subplots(nrows, 2, figsize=(12, 2*nrows), sharex=True)
    axes1 = axes1.flatten()

    for i in range(sdim):
        axes1[i].plot(timesteps, states[:, i], label=state_labels[i])
        axes1[i].set_title(state_labels[i])
        axes1[i].grid(True)
        axes1[i].legend(loc="upper right")

    # Hide unused subplots
    for j in range(sdim, len(axes1)):
        fig1.delaxes(axes1[j])

    axes1[-2].set_xlabel("Seconds")
    axes1[-1].set_xlabel("Seconds")

    fig1.suptitle("State Trajectories", fontsize=14)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    fig1.savefig(f"{save_prefix}_states.png", dpi=300)
    plt.close(fig1)

    # --- Controls + Reward + Contact ---
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for i in range(adim):
        if i == 5:
            axes2[i % 3, 1].step(timesteps, actions[:, i], 
                                where="post", label=action_labels[i], color="orange")
            axes2[i % 3, 1].set_title(plot_titles[i])
            axes2[i % 3, 1].grid(True)
            axes2[i % 3, 1].legend(loc="upper right")
        else:
            axes2[i % 3, 0].step(timesteps, actions[:, i], 
                                where="post", label=action_labels[i])
            axes2[i % 3, 0].set_title(plot_titles[i])
            axes2[i % 3, 0].grid(True)
            axes2[i % 3, 0].legend(loc="upper right")

    # Reward
    axes2[0, 1].plot(timesteps, rewards, label="Reward", color="purple")
    axes2[0, 1].set_title("Reward")
    axes2[0, 1].grid(True)
    axes2[0, 1].legend(loc="upper right")

    # Contact (always last state)
    axes2[1, 1].step(timesteps, states[:, -1], where="post", 
                     label="Contact", color="red")
    axes2[1, 1].set_title("Contact")
    axes2[1, 1].grid(True)
    axes2[1, 1].legend(loc="upper right")

    # Remove unused last subplot
    if adim == 3:
        fig2.delaxes(axes2[2, 1])

    axes2[2, 0].set_xlabel("Seconds")
    axes2[1, 1].set_xlabel("Seconds")
    fig2.suptitle("Controls, Reward, and Contact", fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(f"{save_prefix}_controls.png", dpi=300)
    plt.close(fig2)


def plot_multiple(all_results, dt=0.01):
    # Make sure output folder exists
    os.makedirs("plots", exist_ok=True)

    euler_idx = [0, 1, 2]        # roll, pitch, yaw
    pos_idx = [3, 4, 5]          # x, y, z
    vel_idx = [6, 7, 8]          # vx, vy, vz
    ang_vel_idx = [9, 10, 11]    # wx, wy, wz
    moments_idx = [1, 2, 3]      # mx, my, mz
    forces_idx = [0]      # tx, ty, tz

    groups = {
        "Position": pos_idx,
        "Velocity": vel_idx,
        "Euler Angles": euler_idx,
        "Angular Velocities": ang_vel_idx,
        "Moments": moments_idx,
        "Forces": forces_idx
    }

    for group_name, indices in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns
        axes = axes.flatten()

        for run_idx, run in enumerate(all_results):
            states = run['states']  # shape [T, state_dim]
            actions = run['actions']  # shape [T, action_dim]

            # Compute Euler angles from quaternion if needed
            if states.shape[1] >= 4:
                w = states[:, 0]
                x = states[:, 1]
                y = states[:, 2]
                z = states[:, 3]

                t0 = 2.0 * (w * x + y * z)
                t1 = 1.0 - 2.0 * (x * x + y * y)
                roll = np.arctan2(t0, t1)

                t2 = 2.0 * (w * y - z * x)
                t2 = np.clip(t2, -1.0, 1.0)
                pitch = np.arcsin(t2)

                t3 = 2.0 * (w * z + x * y)
                t4 = 1.0 - 2.0 * (y * y + z * z)
                yaw = np.arctan2(t3, t4)

                euler_angles = np.stack([roll*180/np.pi, pitch*180/np.pi, yaw*180/np.pi], axis=1)
                if states.shape[1] in [8, 14]:
                    states = np.concatenate((euler_angles, states[:, 4:]), axis=1)
                states[:, -3:] = states[:, -3:] * 180/np.pi  # angular velocities to deg/s

            timesteps = np.arange(states.shape[0]) * dt
            if states.size == 0:
                continue

            for i, idx in enumerate(indices):
                if group_name == "Forces" or group_name == "Moments":
                    axes[i].plot(timesteps, actions[:, idx])
                else:
                    axes[i].plot(timesteps, states[:, idx])
                axes[i].set_xlabel('Time [s]')
                axes[i].set_ylabel(f'{group_name}[{i}]')
                axes[i].grid(True)

        plt.suptitle(group_name)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots/{group_name}.png", dpi=300)
        plt.close()



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([100, 100.0, 100], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TeleopLanderCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()