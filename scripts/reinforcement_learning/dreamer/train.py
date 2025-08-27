import argparse
import functools
import os
import pathlib
import sys

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with MBRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-PlanetaryLander-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch"],
    help="The ML framework used for training the MBRL agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="DreamerV3",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "DreamerV3"],
    help="The RL algorithm used for training the MBRL agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""



os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

isaaclab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, isaaclab_root)

from source.DreamerRL import exploration as expl
from source.DreamerRL import models
from source.DreamerRL import tools
from source.DreamerRL.envs import wrappers
from source.DreamerRL.parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

import matplotlib.pyplot as plt
import pickle

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import gymnasium as gym
import random
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
from isaaclab.utils.io import dump_pickle, dump_yaml

from source.DreamerRL.isaaclab_wrapper import IsaacLabToDreamerWrapper, IsaacLabMultiEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

sys.path.append(str(pathlib.Path(__file__).parent))
# Replace {timestamp} in all arguments
for i, arg in enumerate(sys.argv):
    if '{timestamp}' in arg:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sys.argv[i] = arg.replace('{timestamp}', timestamp)


to_np = lambda x: x.detach().cpu().numpy()

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "dreamer_cfg_entry_point" if algorithm in ["dreamerv3"] else f"{algorithm}_cfg_entry_point"


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
        # import gymnasium as gym

        # Create the base IsaacLab environment
        env = gym.make(config.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        
        # Determine observation keys based on your specific IsaacLab environment
        # You may need to customize this based on your environment's observation structure
        obs_keys = getattr(config, 'obs_keys', None)  # Allow configuration of obs keys
        
        # Check if environment has multiple parallel instances
        sample_obs, _ = env.reset()
        
        # If IsaacLab returns multiple environment observations, handle appropriately
        if isinstance(sample_obs, dict):
            # Check if any observation values have batch dimensions
            is_multi_env = False
            num_envs = 1
            
            for v in sample_obs.values():
                if hasattr(v, 'shape') and len(v.shape) > 1:
                    # Assume first dimension is batch/env dimension if > 1
                    potential_num_envs = v.shape[0]
                    if potential_num_envs > 1:
                        is_multi_env = True
                        num_envs = potential_num_envs
                        break
        
        if is_multi_env and num_envs > 1:
            # Handle multi-environment case
            multi_wrapper = IsaacLabMultiEnvWrapper(env, num_envs, obs_keys)
            # Create single environment wrapper for this specific instance
            env = multi_wrapper.create_single_env_wrapper(id % num_envs)
        else:
            # Single environment case
            env = IsaacLabToDreamerWrapper(env, obs_keys)
            
        # Apply standard wrappers
        env = wrappers.NormalizeActions(env)
        
    else:
        # Handle your existing custom environments
        suite, task = config.task.split("_", 1)
        if task == "2dof":
            import envs.lander3 as lander
            env = lander.LanderEnv(task)
            env = wrappers.NormalizeActions(env)

        elif task == "3dofR":
            import envs.lander4 as lander
            env = lander.LanderEnv(task)
            env = wrappers.NormalizeActions(env)

        elif task == "3dofT":
            import envs.lander5 as lander
            env = lander.LanderEnv(task)
            env = wrappers.NormalizeActions(env)
            
        else:
            raise NotImplementedError(f"Unknown task: {task}")
    
    # Apply common wrappers
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env



@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict, dreamer_cfg):
    """Train with MBRL agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    # if args_cli.max_iterations:
    #     agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    # agent_cfg["trainer"]["close_environment_at_exit"] = False

    # randomly sample a seed if seed = -1
    args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    # agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = args_cli.seed

    # specify directory for logging experiments

    tools.set_seed_everywhere(dreamer_cfg.seed)
    if dreamer_cfg.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(dreamer_cfg.logdir).expanduser()
    dreamer_cfg.traindir = dreamer_cfg.traindir or logdir / "train_eps"
    dreamer_cfg.evaldir = dreamer_cfg.evaldir or logdir / "eval_eps"
    dreamer_cfg.steps //= dreamer_cfg.action_repeat
    dreamer_cfg.eval_every //= dreamer_cfg.action_repeat
    dreamer_cfg.log_every //= dreamer_cfg.action_repeat
    dreamer_cfg.time_limit //= dreamer_cfg.action_repeat

    print("Logdir:", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    dreamer_cfg.traindir.mkdir(parents=True, exist_ok=True)
    dreamer_cfg.evaldir.mkdir(parents=True, exist_ok=True)
    with open(logdir / 'dreamer_cfgs.pkl', 'wb') as f:
        pickle.dump(dreamer_cfg,f)
    step = count_steps(dreamer_cfg.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, dreamer_cfg.action_repeat * step)


    # log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.dirname(logdir)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    # set directory into agent config
    # agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    # agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    # resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    # env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    # if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
    #     env = multi_agent_to_single_agent(env)

    # wrap for video recording
    # if args_cli.video:
    #     video_kwargs = {
    #         "video_folder": os.path.join(log_dir, "videos", "train"),
    #         "step_trigger": lambda step: step % args_cli.video_interval == 0,
    #         "video_length": args_cli.video_length,
    #         "disable_logger": True,
    #     }
    #     print("[INFO] Recording videos during training.")
    #     print_dict(video_kwargs, nesting=4)
    #     env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    # env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`





    print("Create envs.")
    if dreamer_cfg.offline_traindir:
        directory = dreamer_cfg.offline_traindir.format(**vars(dreamer_cfg))
    else:
        directory = dreamer_cfg.traindir
    train_eps = tools.load_episodes(directory, limit=dreamer_cfg.dataset_size)
    if dreamer_cfg.offline_evaldir:
        directory = dreamer_cfg.offline_evaldir.format(**vars(dreamer_cfg))
    else:
        directory = dreamer_cfg.evaldir
    eval_eps = tools.load_episodes(directory, limit=5)
    make = lambda mode, id: make_env(env_cfg, dreamer_cfg, mode, id)

    train_envs = [make_env(env_cfg, dreamer_cfg, "train", 0)]

    # train_envs = [make("train", i) for i in range(dreamer_cfg.envs)]
    # eval_envs = [make("eval", i) for i in range(dreamer_cfg.envs)]
    if dreamer_cfg.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        # eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        # train_ens = Damy(train_envs)
        # eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    # acts = train_envs.action_space
    print("Action Space", acts)
    dreamer_cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[1]

    state = None
    if not dreamer_cfg.offline_traindir:
        prefill = max(0, dreamer_cfg.prefill - count_steps(dreamer_cfg.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(dreamer_cfg.num_actions).repeat(dreamer_cfg.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(dreamer_cfg.envs, 1),
                    torch.tensor(acts.high).repeat(dreamer_cfg.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            dreamer_cfg.traindir,
            logger,
            dreamer_cfg,
            limit=dreamer_cfg.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * dreamer_cfg.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, dreamer_cfg)
    eval_dataset = make_dataset(eval_eps, dreamer_cfg)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        dreamer_cfg,
        logger,
        train_dataset,
    ).to(dreamer_cfg.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after dreamer_cfg.steps
    while agent._step < dreamer_cfg.steps + dreamer_cfg.eval_every:
        logger.write()
        # if dreamer_cfg.eval_episode_num > 0:
            # print("Start evaluation.")
            # eval_policy = functools.partial(agent, training=False)
            # tools.simulate(
            #     eval_policy,
            #     eval_envs,
            #     eval_eps,
            #     dreamer_cfg.evaldir,
            #     logger,
            #     dreamer_cfg,
            #     is_eval=True,
            #     episodes=dreamer_cfg.eval_episode_num,
            # )
            # if dreamer_cfg.video_pred_log:
            #     video_pred = agent._wm.video_pred(next(eval_dataset))
            #     logger.video("eval_openl", to_np(video_pred))

            # _, longest_state_dict = max(eval_eps.items(), key=lambda item: len(item[1]['state']))
    
            # state_traj = longest_state_dict["state"]
            # actions = longest_state_dict["action"]
            # xpthrust = np.array([arr[0] for arr in actions])
            # x_thrust = np.array([arr[1] for arr in actions])
            # z_thrust = np.array([arr[2] for arr in actions])
            # # main_thrust = np.array([arr[0] for arr in actions])
            # # f_thrust = np.array([arr[1:5] for arr in actions])
            # # r_thrust = np.array([arr[5:9] for arr in actions])
            # # b_thrust = np.array([arr[9:13] for arr in actions])
            # # l_thrust = np.array([arr[13:17] for arr in actions])
            # x = np.array([arr[0] for arr in state_traj])
            # z = np.array([arr[1] for arr in state_traj])
            # x_dot = np.array([arr[2] for arr in state_traj])
            # z_dot = np.array([arr[3] for arr in state_traj])
            # # quat = np.array([arr[:4] for arr in state_traj])
            # # pos = np.array([arr[4:7] for arr in state_traj])
            # # ang_vel = np.array([arr[7:10] for arr in state_traj])
            # # vel = np.array([arr[10:13] for arr in state_traj])
            # print("Plotting trajectories.")

            # fig = plt.figure(1, figsize=(12, 8))
            # fig.clf()
            # ax = fig.add_subplot(2, 2, 1)
            # ax.plot(x)
            # ax.set_xlabel("Time")
            # ax.set_ylabel("X Position [m]")
            # ax.set_title("Quaternion Trajectory")
            # # ax.legend(["x", "y", "z", "w"])
            # ax = fig.add_subplot(2, 2, 2)
            # ax.plot(z)
            # ax.set_xlabel("Time")
            # ax.set_ylabel("Z Position [m]")
            # ax.set_title("Position Trajectory")
            # # ax.legend(["x", "y", "z"])
            # ax = fig.add_subplot(2, 2, 3)
            # ax.plot(x_dot)
            # ax.set_xlabel("Time")
            # ax.set_ylabel("X Velocity [m/s]")
            # ax.set_title("Velocity Trajectory")
            # # ax.legend(["x", "y", "z"])
            # ax = fig.add_subplot(2, 2, 4)
            # ax.plot(z_dot)
            # ax.set_xlabel("Time")
            # ax.set_ylabel("Z Velocity [m/s]")
            # # ax.legend(["x", "y", "z"])
            # ax.set_title("Angular Velocity Trajectory")
            # plt.tight_layout()
            # plt.show(block=False)
            # plt.pause(1)

            # fig2 = plt.figure(2, figsize=(12, 8))
            # plt.clf()
            # ax2 = fig2.add_subplot(3, 1, 1)
            # ax2.plot(xpthrust)
            # ax2.set_xlabel("Time")
            # ax2.set_ylabel("X + Thrust [N]")
            # ax2.set_title("Main Thrust")
            # ax2 = fig2.add_subplot(3, 1, 2)
            # ax2.plot(x_thrust)
            # ax2.set_xlabel("Time")
            # ax2.set_ylabel(" X - Thrust [N]")
            # ax2.set_title("Forward Thrust")
            # # ax2.legend(["-z", "+y", "+z", "-y"])
            # ax2 = fig2.add_subplot(3, 1, 3)
            # ax2.plot(z_thrust)
            # ax2.set_xlabel("Time")
            # ax2.set_ylabel("Z Thrust [N]")
            # ax2.set_title("Right Thrust")
            # # ax2.legend(["-z", "+x", "+z", "-x"])
            # # ax2 = fig2.add_subplot(3, 2, 4)
            # # ax2.plot(b_thrust)
            # # ax2.set_xlabel("Time")
            # # ax2.set_ylabel("Thrust [N]")
            # # ax2.legend(["-z", "-y", "+z", "+y"])
            # # ax2.set_title("Backward Thrust")
            # # ax2 = fig2.add_subplot(3, 2, 5)
            # # ax2.plot(l_thrust)
            # # ax2.set_xlabel("Time")
            # # ax2.set_ylabel("Thrust [N]")
            # # ax2.legend(["-z", "-x", "+z", "+x"])
            # # ax2.set_title("Left Thrust")
            # plt.tight_layout()
            # plt.show(block=False)
            # plt.pause(1)

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            dreamer_cfg.traindir,
            logger,
            dreamer_cfg,
            limit=dreamer_cfg.dataset_size,
            steps=dreamer_cfg.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs: # + eval_envs:
        try:
            env.close()
        except Exception:
            pass


























































# isaaclab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# sys.path.insert(0, isaaclab_root)

# from source.lander_assets.lander_vehicle_rgd import LUNAR_LANDER_CFG


# @configclass
# class CartpoleSceneCfg(InteractiveSceneCfg):
#     """Configuration for a cart-pole scene."""

#     # ground plane
#     ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
#                           spawn=sim_utils.UsdFileCfg(usd_path=f"/workspace/isaaclab/source/lander_assets/moon_terrain_smooth.usd")
#                           )
#     # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


#     # lights
#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )

#     # articulation
#     # lander: ArticulationCfg = LUNAR_LANDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#     lander: RigidObjectCfg = LUNAR_LANDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#     # sensors
#     camera = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/MainBody/front_cam",
#         update_period=0.01,
#         height=720,
#         width=1280,
#         data_types=["rgb", "distance_to_image_plane"],
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
#         ),
#         offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -2.03), rot=rot_utils.euler_angles_to_quats(np.array([-90, 90, 0]), degrees=True), convention="world"),
#     )
#     height_scanner = RayCasterCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/MainBody",
#         update_period=0.02,
#         offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -2.03)),
#         attach_yaw_only=True,
#         pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
#         debug_vis=False,
#         mesh_prim_paths=["/World/defaultGroundPlane/Mountain_4"],
#     )
#     contact_forces = ContactSensorCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/MainBody", 
#         update_period=0.0, 
#         track_air_time = True,
#         debug_vis=True,
#         history_length=5,
#         filter_prim_paths_expr=["/World/defaultGroundPlane/Mountain_4"]
#     )
#     imu = ImuCfg(
#         prim_path="{ENV_REGEX_NS}/Robot/MainBody",
#         update_period=0.0,
#         offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
#         debug_vis=True,
#         gravity_bias=(0, 0, 0),
#     )


# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     """Runs the simulation loop."""
#     # Extract scene entities
#     # note: we only do this here for readability.
#     robot = scene["lander"]
#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     count = 0
#     height = 1e1
#     # Simulation loop

#     while simulation_app.is_running():
#         # Reset
#         # print(scene["contact_forces"])
#         if scene["contact_forces"].data.current_contact_time[0] > 1.0 or count > 1000:
#             # print(scene["contact_forces"].data.current_contact_time[0])
#             # reset counter
#             print(torch.Tensor(robot.root_physx_view.get_masses()))
#             print(torch.Tensor(robot.root_physx_view.get_masses().sum()))
#             count = 0
#             # reset the scene entities
#             # root state
#             # we offset the root state by the origin since the states are written in simulation world frame
#             # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
#             root_state = robot.data.default_root_state.clone()
#             root_state[:, :3] += scene.env_origins
#             robot.write_root_pose_to_sim(root_state[:, :7])
#             robot.write_root_velocity_to_sim(root_state[:, 7:])
#             # set joint positions with some noise
#             # joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
#             # joint_pos += torch.rand_like(joint_pos) * 0.1
#             # robot.write_joint_state_to_sim(joint_pos, joint_vel)
#             # clear internal buffers
#             scene.reset()
#             print("[INFO]: Resetting robot state...")
#         # Apply random action
#         # -- generate random joint efforts
#         # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
#         # -- apply action to the robot
#         base_gravity = torch.Tensor(sim.cfg.gravity)
        
#         mass = torch.Tensor(robot.root_physx_view.get_masses()[0])
#         robot.set_external_force_and_torque(
#                 forces=0*mass*base_gravity.expand(scene.num_envs, 1, 3).clone(),
#                 torques=torch.zeros(scene.num_envs,1,3),
#             )
#         # robot.set_joint_effort_target(efforts)
#         # -- write data to sim
#         scene.write_data_to_sim()
#         # Perform step
#         sim.step()
#         # Increment counter
#         height = torch.max(scene["height_scanner"].data.pos_w[..., -1]).item() - torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item() - 2.03

#         if count % 10 == 0 and count > 0:
#             # img_data =scene["camera"].data.output["rgb"][0,:,:,:].cpu()
#             # plt.imshow(img_data)
#             # plt.axis('off')
#             # plt.savefig("/workspace/isaaclab/source/test_images/my_plot.png") 
#             print("Sensor 1 Received contact force of: ", scene["contact_forces"].data.net_forces_w_history[0,:,:,:])
#             print("Sensor 2 Received contact force of: ", scene["contact_forces"].data.net_forces_w_history[1,:,:,:])
#             print("Sensor shape: ", scene["contact_forces"].data.net_forces_w_history[0,:,:,:].shape)
#             # print("Ray Caster Height: ", scene["height_scanner"].data.pos_w[0, :])
#             # print("Ground plane height: ", scene["height_scanner"].data.ray_hits_w[0,0, :])
#             # print("Received max height value: ", height)
#             # print("IMU Linear Acceleration: ", scene["imu"].data.lin_acc_b[0])
#             # print("IMU Angular Acceleration: ", scene["imu"].data.ang_acc_b[0])
#         count += 1
#         # Update buffers
#         scene.update(sim_dt)





# def main():
#     """Main function."""
#     # Load kit helper
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim = SimulationContext(sim_cfg)
#     # Set main camera
#     sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
#     # Design scene
#     scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=50.0)
#     scene = InteractiveScene(scene_cfg)
#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     run_simulator(sim, scene)
    

if __name__ == "__main__":
    # run the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")

    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[3] / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "lander" / "agents" / "dreamer_cfg.yaml").read_text()
    )
    configs["defaults"]["logdir"] = f"logs/IsaacLab/lander_direct/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(dreamer_cfg=parser.parse_args(remaining)) # type: ignore
    # main()
    # close sim app
    simulation_app.close()