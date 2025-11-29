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

import pygame

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


    pygame.init()
    pygame.joystick.init()

    # if pygame.joystick.get_count() == 0:
    #     print("No joystick connected.")
    #     return
    joystick = pygame.joystick.Joystick(0)

    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")

    

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

            pygame.event.pump()  # Process event queue

            main_engine = unnormalize(joystick.get_axis(5),0,43000) # -1 to 1
            pitch = unnormalize(joystick.get_axis(1),-400,400)  # -1 to 1
            roll = unnormalize(joystick.get_axis(0),-400,400)   # -1 to 1
            yaw = unnormalize(joystick.get_axis(3),-400,400)    # -1 to 1

            print(f"""
                    Main Engine: {main_engine:.2f}, 
                    Pitch: {pitch:.2f}, 
                    Roll: {roll:.2f}, 
                    Yaw: {yaw:.2f}
                """)

            action[0]['action'] = np.array([main_engine, pitch, roll, yaw])
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

    # now all_results is a list of 100 dicts, each containing the simulation history

    
    # if len(state_hist) > 1:
    #     state_traj = np.stack(state_hist)
    #     timesteps = np.arange(state_traj.shape[0])
    #     plot_trajectory(state_traj, actions, rewards, timesteps, dt=dt, save_prefix="traj")
 
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
    elif args_cli.task.startswith('Isaac-Cartpole-Direct-'):
        task = "cartpole_direct"
    elif args_cli.task.startswith('Isaac-Cartpole-RGB-Camera-Direct-'):
        task = "cartpole_camera_direct"
    log_root_path = os.path.join("logs", "IsaacLab", task)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    resume_path = get_checkpoint_path(log_root_path) # gets last run
    # resume_path = os.path.join("20251113_043058",log_root_path)
    log_dir = os.path.dirname(resume_path)
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