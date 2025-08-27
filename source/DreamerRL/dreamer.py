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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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

sys.path.append(str(pathlib.Path(__file__).parent))

import source.DreamerRL.exploration as expl
import source.DreamerRL.models as models
import source.DreamerRL.tools as tools
import source.DreamerRL.envs.wrappers as wrappers
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

# import omni.isaac.lab_tasks  # This registers IsaacLab environments
from isaaclab_wrapper import IsaacLabToDreamerWrapper, IsaacLabMultiEnvWrapper

import datetime
# Replace {timestamp} in all arguments
for i, arg in enumerate(sys.argv):
    if '{timestamp}' in arg:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        sys.argv[i] = arg.replace('{timestamp}', timestamp)


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
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
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
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
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
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
        if self._config.expl_behavior != "greedy":
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


# def make_env(config, mode, id):
#     suite, task = config.task.split("_", 1)
#     if task == "2dof":
#         import envs.lander3 as lander
#         env = lander.LanderEnv(task)
#         env = wrappers.NormalizeActions(env)

#     elif task == "3dofR":
#         import envs.lander4 as lander
#         env = lander.LanderEnv(task)
#         env = wrappers.NormalizeActions(env)

#     elif task == "3dofT":
#         import envs.lander5 as lander
#         env = lander.LanderEnv(task)
#         env = wrappers.NormalizeActions(env)
        
#     else:
#         raise NotImplementedError(suite)
#     env = wrappers.TimeLimit(env, config.time_limit)
#     env = wrappers.SelectAction(env, key="action")
#     env = wrappers.UUID(env)
#     return env


# Replace the make_env function with this version:
def make_env(config, mode, id):
    """Create environment with IsaacLab support."""
    
    env_cfg =  DirectRLEnvCfg
    agent_cfg = {}
    
    # Check if it's an IsaacLab environment
    if config.task.startswith('Isaac-'):
        # Create IsaacLab environment
        import gymnasium as gym
        
        # config shortcuts
        algorithm = args_cli.algorithm.lower()
        agent_cfg_entry_point = "dreamer_cfg_entry_point" if algorithm in ["dreamerv3"] else f"{algorithm}_cfg_entry_point"

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





def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    with open(logdir / 'configs.pkl', 'wb') as f:
        pickle.dump(config,f)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=5)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
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
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

            _, longest_state_dict = max(eval_eps.items(), key=lambda item: len(item[1]['state']))
    
            state_traj = longest_state_dict["state"]
            actions = longest_state_dict["action"]
            xpthrust = np.array([arr[0] for arr in actions])
            x_thrust = np.array([arr[1] for arr in actions])
            z_thrust = np.array([arr[2] for arr in actions])
            # main_thrust = np.array([arr[0] for arr in actions])
            # f_thrust = np.array([arr[1:5] for arr in actions])
            # r_thrust = np.array([arr[5:9] for arr in actions])
            # b_thrust = np.array([arr[9:13] for arr in actions])
            # l_thrust = np.array([arr[13:17] for arr in actions])
            x = np.array([arr[0] for arr in state_traj])
            z = np.array([arr[1] for arr in state_traj])
            x_dot = np.array([arr[2] for arr in state_traj])
            z_dot = np.array([arr[3] for arr in state_traj])
            # quat = np.array([arr[:4] for arr in state_traj])
            # pos = np.array([arr[4:7] for arr in state_traj])
            # ang_vel = np.array([arr[7:10] for arr in state_traj])
            # vel = np.array([arr[10:13] for arr in state_traj])
            print("Plotting trajectories.")

            fig = plt.figure(1, figsize=(12, 8))
            fig.clf()
            ax = fig.add_subplot(2, 2, 1)
            ax.plot(x)
            ax.set_xlabel("Time")
            ax.set_ylabel("X Position [m]")
            ax.set_title("Quaternion Trajectory")
            # ax.legend(["x", "y", "z", "w"])
            ax = fig.add_subplot(2, 2, 2)
            ax.plot(z)
            ax.set_xlabel("Time")
            ax.set_ylabel("Z Position [m]")
            ax.set_title("Position Trajectory")
            # ax.legend(["x", "y", "z"])
            ax = fig.add_subplot(2, 2, 3)
            ax.plot(x_dot)
            ax.set_xlabel("Time")
            ax.set_ylabel("X Velocity [m/s]")
            ax.set_title("Velocity Trajectory")
            # ax.legend(["x", "y", "z"])
            ax = fig.add_subplot(2, 2, 4)
            ax.plot(z_dot)
            ax.set_xlabel("Time")
            ax.set_ylabel("Z Velocity [m/s]")
            # ax.legend(["x", "y", "z"])
            ax.set_title("Angular Velocity Trajectory")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)

            fig2 = plt.figure(2, figsize=(12, 8))
            plt.clf()
            ax2 = fig2.add_subplot(3, 1, 1)
            ax2.plot(xpthrust)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("X + Thrust [N]")
            ax2.set_title("Main Thrust")
            ax2 = fig2.add_subplot(3, 1, 2)
            ax2.plot(x_thrust)
            ax2.set_xlabel("Time")
            ax2.set_ylabel(" X - Thrust [N]")
            ax2.set_title("Forward Thrust")
            # ax2.legend(["-z", "+y", "+z", "-y"])
            ax2 = fig2.add_subplot(3, 1, 3)
            ax2.plot(z_thrust)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Z Thrust [N]")
            ax2.set_title("Right Thrust")
            # ax2.legend(["-z", "+x", "+z", "-x"])
            # ax2 = fig2.add_subplot(3, 2, 4)
            # ax2.plot(b_thrust)
            # ax2.set_xlabel("Time")
            # ax2.set_ylabel("Thrust [N]")
            # ax2.legend(["-z", "-y", "+z", "+y"])
            # ax2.set_title("Backward Thrust")
            # ax2 = fig2.add_subplot(3, 2, 5)
            # ax2.plot(l_thrust)
            # ax2.set_xlabel("Time")
            # ax2.set_ylabel("Thrust [N]")
            # ax2.legend(["-z", "-x", "+z", "+x"])
            # ax2.set_title("Left Thrust")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

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
    main(parser.parse_args(remaining))
