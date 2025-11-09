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
from source.DreamerRL import safe_models
from source.DreamerRL import tools
from source.DreamerRL.envs import wrappers
from source.DreamerRL.parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
import psutil

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
import time

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
from isaaclab.utils.io import dump_yaml

from source.DreamerRL.isaaclab_wrapper import IsaacLabDreamerWrapper

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
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean() + self._wm.heads["safety"](f).mean() # modified for safety
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
        reward = lambda f, s, a: (
                                  self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode() +
                                  self._wm.heads["safety"](self._wm.dynamics.get_feat(s)).mode()
                                  )  # modified for safety 
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
        
        env = IsaacLabDreamerWrapper(env, obs_keys)
            
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
    # make = lambda mode, id: make_env(env_cfg, dreamer_cfg, mode, id)

    train_envs = [make_env(env_cfg, dreamer_cfg, "train", 0)]

    # making place holder envs for dreamer 
    import source.DreamerRL.envs.lander5 as lander
    for i in range(1,dreamer_cfg.envs):
        env = lander.LanderEnv("3dofT")
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, dreamer_cfg.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        train_envs.append(env)


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

if __name__ == "__main__":
    start_time = time.time()
    # run the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")

    args, remaining = parser.parse_known_args()
    if args_cli.task.startswith('Isaac-PlanetaryLander-Direct-States-'):
        cfg_name = "dreamer_states_cfg.yaml"
        exp_name = "lander_states_direct"
    elif args_cli.task.startswith('Isaac-PlanetaryLander-Direct-6DOF-'):
        cfg_name = "dreamer_6dof_cfg.yaml"
        exp_name = "lander_6dof_direct"
    elif args_cli.task.startswith('Isaac-Cartpole-RGB-Camera-Direct-'):
        cfg_name = "dreamer_camera_cfg.yaml"
        exp_name = "cartpole_camera_direct"
    elif args_cli.task.startswith('Isaac-Cartpole-Direct-'):
        cfg_name = "dreamer_cfg.yaml"
        exp_name = "cartpole_direct"
    else:
        cfg_name = "dreamer_cfg.yaml"
        exp_name = "lander_direct"

    configs = yaml.safe_load(
            (pathlib.Path(__file__).resolve().parents[3] / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / exp_name.split('_', 1)[0] / "agents" / cfg_name).read_text()
        )
    configs["defaults"]["logdir"] = f"logs/IsaacLab/{exp_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # recursive update function                                 
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

    end_time = time.time()
    elapsed = end_time - start_time  # seconds

    # Convert to hrs:min:sec
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    # close sim app
    simulation_app.close()