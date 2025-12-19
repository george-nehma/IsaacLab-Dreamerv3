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

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=True, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="DreamerV3",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "DreamerV3"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
# args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
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

from packaging import version

os.environ["MUJOCO_GL"] = "osmesa"

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

from datetime import datetime
from packaging import version

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml

from source.DreamerRL.isaaclab_wrapper import IsaacLabDreamerWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, dreamer_cfg, logger, dataset):
        super(Dreamer, self).__init__()
        self._dreamer_cfg = dreamer_cfg
        self._logger = logger
        self._should_log = tools.Every(dreamer_cfg.log_every)
        batch_steps = dreamer_cfg.batch_size * dreamer_cfg.batch_length
        self._should_train = tools.Every(batch_steps / dreamer_cfg.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(dreamer_cfg.reset_every)
        self._should_expl = tools.Until(int(dreamer_cfg.expl_until / dreamer_cfg.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // dreamer_cfg.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, dreamer_cfg)
        self._task_behavior = models.ImagBehavior(dreamer_cfg, self._wm)
        if (
            dreamer_cfg.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(dreamer_cfg, act_space),
            plan2explore=lambda: expl.Plan2Explore(dreamer_cfg, self._wm, reward),
        )[dreamer_cfg.expl_behavior]().to(self._dreamer_cfg.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._dreamer_cfg.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._dreamer_cfg.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._dreamer_cfg.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._dreamer_cfg.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._dreamer_cfg.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._dreamer_cfg.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._dreamer_cfg.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "dreamer_cfg_entry_point" if algorithm in ["dreamerv3"] else f"{algorithm}_cfg_entry_point"

to_np = lambda x: x.detach().cpu().numpy()

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

# Replace the make_env function with this version:
def make_env(env_cfg, config, mode, id):
    """Create environment with IsaacLab support."""
    
    # Check if it's an IsaacLab environment
    if config.task.startswith('Isaac-'):
        # Create IsaacLab environment

        # Create the base IsaacLab environment
        env = gym.make(config.task, cfg=env_cfg, render_mode="rgb_array" if mode=="video" else None)
        
        # wrap for video recording
        if mode == "video":
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Determine observation keys based on your specific IsaacLab environment
        # You may need to customize this based on your environment's observation structure
        obs_keys = getattr(config, 'obs_keys', None)  # Allow configuration of obs keys
        
        env = IsaacLabDreamerWrapper(env, obs_keys)
            
        # Apply standard wrappers
        env = wrappers.NormalizeActions(env)
    
    # Apply common wrappers
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env

import matplotlib.pyplot as plt
import numpy as np

def plot_multiple(all_results, dt=0.1, center=np.array([0, 0])):
    # Make sure output folder exists
    os.makedirs("plots", exist_ok=True)


    euler_idx = [0, 1, 2]        # roll, pitch, yaw
    pos_idx = [3, 4, 5]          # x, y, z
    vel_idx = [6, 7, 8]          # vx, vy, vz
    ang_vel_idx = [9, 10, 11]    # wx, wy, wz
    moments_idx = [3, 4, 5]      # mx, my, mz
    forces_idx = [0, 1, 2]      # tx, ty, tz
    rewards_idx = [0]            # rewards

    groups = {
        "Position": pos_idx,
        "Velocity": vel_idx,
        "Euler Angles": euler_idx,
        "Angular Velocities": ang_vel_idx,
        "Moments": moments_idx,
        "Forces": forces_idx,
        "Rewards": rewards_idx,
        "3D Plot": pos_idx,
    }

    for group_name, indices in groups.items():

        if group_name == "3D Plot":
            fig = plt.figure(figsize=(4, 6)) 
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect((1, 1, 3))
            ax.view_init(elev=30, azim=45)
            plt.tight_layout()

            for run_idx, run in enumerate(all_results):
                states = run['states']  # shape [T, state_dim]
                if states.shape[0] > 500:
                    continue
                # Extract positions
                x = states[:, 4]
                y = states[:, 5]
                z = states[:, 6]

                ax.plot(x + center[0], y + center[1], z, label=f'Run {run_idx+1}')

            ax.locator_params(axis='x', nbins=3)
            ax.locator_params(axis='y', nbins=3)
            # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            #     axis.pane.set_visible(False)
            #     axis.set_tick_params(pad=6)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.zaxis.set_rotate_label(False) 
            ax.set_zlabel('Z [m]', rotation=90)
            # ax.set_title('3D Trajectory Plot')
            ax.set_ylim([center[1]-30, center[1]+30])
            ax.set_xlim([center[0]-30, center[0]+30])
            plt.savefig(f"plots/3D_Trajectory.png", dpi=300)
            
            plt.close()
            continue
        elif group_name == "Rewards":
            fig, ax = plt.subplots(figsize=(8, 6))
            for run_idx, run in enumerate(all_results):
                rewards = run['rewards']  # shape [T, ]
                if rewards.shape[0] > 500:
                    continue
                timesteps = np.arange(rewards.shape[0]) * dt
                ax.plot(timesteps[0:-1], rewards[0:-1], label=f'Run {run_idx+1}')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(f'{group_name}')
                ax.grid(True)
            # plt.suptitle(group_name)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"plots/{group_name}.png", dpi=300)
            plt.close()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns
            axes = axes.flatten()

            for run_idx, run in enumerate(all_results):
                states = run['states']  # shape [T, state_dim]
                if states.shape[0] > 500:
                    continue
                # states[:, 4] += 10  # shift x
                # states[:, 5] += -10  # shift y
                actions = run['actions']  # shape [T, action_dim]
                rewards = run['rewards']  # shape [T, ]

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

                for i, idx in enumerate(indices):
                    axis = 'X' if i == 0 else 'Y' if i == 1 else 'Z'
                    unit = '[m]' if group_name == "Position" else '[m/s]' if group_name == "Velocity" else '[deg]' if group_name == "Euler Angles" else '[deg/s]' if group_name == "Angular Velocities" else '[N]' if group_name == "Forces" else '[Nm]' if group_name == "Moments" else ''
                    if group_name == "Forces" or group_name == "Moments":
                        axes[i].plot(timesteps, actions[:, idx])
                    # elif group_name == "Rewards":
                    #     axes[i].plot(timesteps, rewards)
                    else:
                        axes[i].plot(timesteps, states[:, idx])
                    axes[i].set_xlabel('Time [s]')
                    axes[i].set_ylabel(f'{axis} {group_name} {unit}')
                    axes[i].grid(True)

            # plt.suptitle(group_name)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"plots/{group_name}.png", dpi=300)
            plt.close()

def plot_landing(all_results, dt=0.1, center=np.array([0, 0])):
    import matplotlib.patches as patches

    pos_idx = [4, 5]   # x, y

    # Collect all landing positions to compute axis limits
    # landing_points = []
    # for run in all_results:
    #     loc = run["states"][-1, pos_idx] + center  # shift landing zone center to (10,10)
    #     landing_points.append(loc)

    # landing_points = np.array(landing_points)
    # max_val = float(np.max(np.abs(landing_points)))   # highest |x| or |y|
    # axis_limit = max_val + 0.5         # always ≥3, add margin

    fig, ax = plt.subplots(figsize=(8, 6))

    g_count = 0
    y_count = 0
    o_count = 0
    r_count = 0
    for run_idx, run in enumerate(all_results):
        # if run["states"].size > 500:
        #             continue
        loc = run["states"][-1, pos_idx]
        if np.linalg.norm(run["states"][-1,7:9]) > 0.3 and np.linalg.norm(loc) < 2.0:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='o', color='orange',fillstyle='none', markersize=8, markeredgewidth=1.5)
            print(f"Run {run_idx+1} - X Velocity: {run['states'][-1,7]:.3f} m/s, Y Velocity: {run['states'][-1,8]:.3f} m/s, Z Velocity: {run['states'][-1,9]:.3f} m/s, Norm Velocity: {np.linalg.norm(run['states'][-1,7:9]):.3f} m/s")
            o_count += 1
        elif np.linalg.norm(run["states"][-1,7:9]) < 0.3 and np.linalg.norm(loc) > 2.0:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='o', color='yellow',fillstyle='none', markersize=8, markeredgewidth=1.5)
            y_count += 1
        elif np.linalg.norm(run["states"][-1,7:9]) > 0.3 and  np.linalg.norm(loc) > 2.0:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='x', color='red')
            r_count += 1
        else:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='o', color='green',fillstyle='none', markersize=8, markeredgewidth=1.5)
            g_count += 1
    print(f"Green: {g_count}, Yellow: {y_count}, Orange: {o_count}, Red: {r_count}")

    # Target site
    ax.plot(center[0], center[1], 'b*', markersize=15)

    # Light grey landing-zone circle
    landing_zone = patches.Circle(
        (center[0],center[1]),
        radius=2.0,
        linewidth=1,
        alpha=0.5,
        color='lightgrey',
    )
    ax.add_patch(landing_zone)

    # Dotted reference lines at X=0 and Y=0
    ax.axhline(center[1], color='black', linestyle=':', linewidth=1)
    ax.axvline(center[0], color='black', linestyle=':', linewidth=1)

    # Axes limits (auto-expanded but minimum ±3)
    ax.set_xlim(left=center[0]-5, right=center[0]+5)
    ax.set_ylim(bottom=center[1]-5, top=center[1]+5)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')

    plt.savefig("plots/Landing_Zone.png", dpi=300)
    plt.close()
    



@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict, dreamer_cfg):
    """Train with MBRL agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    # if args_cli.distributed:
    #     env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"


    # randomly sample a seed if seed = -1
    args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = "cpu"
    env_cfg.sim.use_fabric = False

    logdir = pathlib.Path(dreamer_cfg.logdir).expanduser()
    print("Logdir:", logdir)

    step = count_steps(dreamer_cfg.traindir)
    logger = tools.Logger(logdir, dreamer_cfg.action_repeat * step)

    print("Create envs.")
    test_env = make_env(env_cfg,dreamer_cfg, "no-video", 0)
    test_env = Damy(test_env)
    directory = dreamer_cfg.traindir
    test_eps = tools.load_episodes(directory, limit=dreamer_cfg.dataset_size)
    test_dataset = make_dataset(test_eps, dreamer_cfg)

    acts = test_env.action_space
    print("Action Space", acts)
    dreamer_cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[1]

    agent = Dreamer(
            test_env.observation_space,
            test_env.action_space,
            dreamer_cfg,
            logger,
            test_dataset,
        ).to(dreamer_cfg.device)
    agent.requires_grad_(requires_grad=False)

    if (logdir / "latest.pt").exists():
        print("Loading latest checkpoint...")
        checkpoint = torch.load(logdir / "latest.pt", weights_only=True)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    def convert(value, precision=32):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        elif np.issubdtype(value.dtype, bool):
            dtype = bool
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)

    def unnormalize(x_norm, min_val, max_val):
        return 0.5 * (x_norm + 1) * (max_val - min_val) + min_val

    # get environment (step) dt for real-time evaluation
    try:
        dt = test_env.step_dt
    except AttributeError:
        dt = test_env.unwrapped.step_dt

    num_runs = 5
    all_results = []  # will store results of all simulations

    for run_idx in range(num_runs):
        done = np.ones(1, bool)
        obs = [None]
        # reset environment
        r = test_env.reset()
        result = r()
        obs[0] = result
        action_low = test_env.env.env.env.env.action_space.low[0]
        action_high = test_env.env.env.env.env.action_space.high[0]
        # action_prev = np.zeros_like(test_env.env.env.env.env.action_space)
        # alpha = 0.5
        timestep = 0
        agent_state = None

        # per-run histories
        state_hist = []
        control_hist = []
        reward_hist = []

        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()

            # run everything in inference mode
            
            # agent stepping
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
            with torch.inference_mode():
                action, agent_state = agent(obs, done, agent_state, training=False)
            if isinstance(action, dict):
                action = [
                    {k: np.array(action[k][0].detach().cpu()) for k in action}
                ]
            else:
                action = np.array(action)
            # action[0]['action'] = alpha*action_prev + (1-alpha)*action[0]['action']
            r = test_env.step(action)
            results = r()
            if not isinstance(results[1], float):
                test = []
                for i in range(1):  # only one env
                    tuple_entry = []
                    for elem in results:
                        if isinstance(elem, dict):
                            tuple_entry.append(elem)
                        else:
                            tuple_entry.append(elem[i])
                    test.append(tuple(tuple_entry))
                results = test

            obs, reward, done = zip(*[p[:3] for p in results])
            done = np.array(done)
            reward_hist.append(reward[0])
            if 'state' in obs[0]:
                state_hist.append(obs[0]['state'])

            control_hist.append(unnormalize(action[0]['action'], action_low, action_high))

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            # real-time sleep
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

            if done[0]:
                break

        # store results for this run
        run_results = {
            'states': np.stack(state_hist) if state_hist else np.array([]),
            'actions': np.stack(control_hist) if control_hist else np.array([]),
            'rewards': np.array(reward_hist)
        }
        all_results.append(run_results)

    torch.save(all_results, os.path.join(logdir, f"play_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"))

    center=np.array([10, -5])
    plot_multiple(all_results, dt, center)
    plot_landing(all_results, dt, center)
    # close the simulator
    test_env.close()



if __name__ == "__main__":
    # run the main function
    # specify directory for logging experiments (load checkpoint)

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")

    args, remaining = parser.parse_known_args()
    if args_cli.task.startswith('Isaac-PlanetaryLander-Direct-States-'):
        task = "lander_states_direct"
    elif args_cli.task.startswith('Isaac-PlanetaryLander-Direct-6DOF-'):
        task = "lander_6dof_direct"
    elif args_cli.task.startswith('Isaac-PlanetaryLander-Direct-'):
        task = "lander_direct"
    elif args_cli.task.startswith('Isaac-Cartpole-Direct-'):
        task = "cartpole_direct"
    elif args_cli.task.startswith('Isaac-Cartpole-RGB-Camera-Direct-'):
        task = "cartpole_camera_direct"
    log_root_path = os.path.join("logs", "IsaacLab", task)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint is None:
        resume_path = get_checkpoint_path(log_root_path) # gets last run
        log_dir = os.path.dirname(resume_path)
    else:
        log_dir = os.path.join(log_root_path, args_cli.checkpoint)
    
    cfg_path = os.path.join(log_dir, "dreamer_cfgs.pkl")
    with open(cfg_path, 'rb') as f:
        config = pickle.load(f)
        # config.parallel = False
        config.time_limit = 15000

    print(config.eval_every)
    
    main(dreamer_cfg=config) # type: ignore
    # main()
    # close sim app
    simulation_app.close()