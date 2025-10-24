import numpy as np
import torch
import gymnasium as gym

to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class IsaacLabDreamerWrapper(gym.Wrapper):
    """
    Unified wrapper for IsaacLab environments (single or multi-env) to DreamerV3 format.
    """

    def __init__(self, env, obs_keys=None):
        super().__init__(env)
        self.env = env
        self.obs_keys = obs_keys

        # Detect number of parallel envs
        # sample_obs, _ = self.env.reset()
        self.num_envs = env.env.num_envs

        # Setup observation space
        self._setup_observation_space()

        self._episode_step = np.zeros(self.num_envs, dtype=np.int32)
        self._max_episode_steps = getattr(env, "_max_episode_steps", 5000)

    def _detect_num_envs(self, sample_obs):
        """Figure out how many parallel envs there are."""
        if isinstance(sample_obs, dict):
            for v in sample_obs.values():
                if hasattr(v, "shape") and len(v.shape) > 1:
                    return v.shape[0]
        elif hasattr(sample_obs, "shape") and len(sample_obs.shape) > 1:
            return sample_obs.shape[0]
        return 1

    def _setup_observation_space(self, ):
        """Setup Dreamer-style observation space."""
        state_space = self.observation_space.shape[-1]
        spaces = {
            "state": gym.spaces.Box(dtype=np.float32, shape=(state_space,), low=-np.inf, high=np.inf),
            "reward": gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
            "is_first": gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
            "is_last": gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
            "is_terminal": gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
            "log/reward": gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
        }

        if "image" in self.obs_keys:
            image_shape = (self.env.env.cfg.pix_x, self.env.env.cfg.pix_y) + (3,)
            spaces = {"image": gym.spaces.Box(dtype=np.float32, shape=image_shape, low=0, high=255), **spaces}

        self.observation_space = gym.spaces.Dict(spaces)

    def _process_observation(self, obs):
        """Convert IsaacLab observation(s) into Dreamer dict(s)."""
        if isinstance(obs, dict):
            # If multi-env, flatten per env
            if self.num_envs > 1:
                processed = []
                for i in range(self.num_envs):
                    processed.append({k: to_np(v[i]) for k, v in obs.items()})
                return processed
            else:
                return {k: to_np(v[0]) if hasattr(v, "__getitem__") and np.ndim(v) > 0 else to_np(v)
                        for k, v in obs.items()}
        else:
            return to_np(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_step[:] = 0
        obs['is_first'] = self.env.env.firsts
        processed_obs = self._process_observation(obs)

        return processed_obs #, dreamer_info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.env.env.device)

        obs, reward, done, info = self.env.step(action)
        processed_obs = self._process_observation(obs)

        self._episode_step += 1

        # Handle reward
        reward = to_np(reward)
        done = to_np(done)

        return processed_obs, reward, done, info

    def close(self):
        self.env.close()