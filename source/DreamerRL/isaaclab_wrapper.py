import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional
import copy

to_np = lambda x: x.detach().cpu().numpy()


class IsaacLabToDreamerWrapper(gym.Wrapper):
    """
    Wrapper to adapt IsaacLab environments for use with DreamerV3.
    
    This wrapper handles:
    - Converting IsaacLab observations to Dreamer-compatible format
    - Managing episode resets and termination flags
    - Ensuring observations are properly shaped for Dreamer's world model
    """
    
    def __init__(self, env, obs_keys=None):
        """
        Args:
            env: IsaacLab environment instance
            obs_keys: List of observation keys to extract from IsaacLab obs dict.
                     If None, will try to automatically detect relevant keys.
        """
        super().__init__(env)
        self.env = env
        self.obs_keys = obs_keys
        
        # Store initial observation to determine structure
        self._setup_observation_space()
        
        # Track episode state
        self._episode_step = 0
        self._max_episode_steps = getattr(env, '_max_episode_steps', 5000)
        
    def _setup_observation_space(self):
        """Setup observation space based on IsaacLab environment."""
        # Get a sample observation to understand structure
        sample_obs, _ = self.env.reset()
        
        if self.obs_keys is None:
            # Auto-detect relevant observation keys
            if isinstance(sample_obs, dict):
                # Common IsaacLab observation keys
                potential_keys = ['policy', 'state', 'observations', 'obs']
                self.obs_keys = []
                
                for key in potential_keys:
                    if key in sample_obs:
                        self.obs_keys.append(key)
                        break
                
                # If no standard keys found, use all keys
                if not self.obs_keys:
                    self.obs_keys = list(sample_obs.keys())
            else:
                # If observation is not a dict, use it directly
                self.obs_keys = None
        
        # Create flattened observation space
        self._create_observation_space(sample_obs)
        
    def _create_observation_space(self, sample_obs):
        """Create a flattened observation space for Dreamer."""
        state_space = sample_obs["state"].shape[1]
        image_space = sample_obs["image"][0].shape

        # if isinstance(sample_obs, dict):
        #     if self.obs_keys:
        #         # Extract specified keys and flatten
        #         obs_parts = []
        #         for key in self.obs_keys:
        #             if key in sample_obs:
        #                 obs_part = sample_obs[key]
        #                 if isinstance(obs_part, torch.Tensor):
        #                     obs_part = obs_part.cpu().numpy()
        #                 obs_parts.append(obs_part.flatten())
                
        #         if obs_parts:
        #             flattened_obs = np.concatenate(obs_parts)
        #         else:
        #             # Fallback: flatten all values
        #             obs_parts = []
        #             for value in sample_obs.values():
        #                 if isinstance(value, torch.Tensor):
        #                     value = value.cpu().numpy()
        #                 if isinstance(value, np.ndarray):
        #                     obs_parts.append(value.flatten())
        #             flattened_obs = np.concatenate(obs_parts) if obs_parts else np.array([0.0])
        #     else:
        #         # Flatten all observation values
        #         obs_parts = []
        #         for value in sample_obs.values():
        #             if isinstance(value, torch.Tensor):
        #                 value = value.cpu().numpy()
        #             if isinstance(value, np.ndarray):
        #                 obs_parts.append(value.flatten())
        #         flattened_obs = np.concatenate(obs_parts) if obs_parts else np.array([0.0])
        # else:
        #     # Direct observation
        #     if isinstance(sample_obs, torch.Tensor):
        #         sample_obs = sample_obs.cpu().numpy()
        #     flattened_obs = sample_obs.flatten()
        
        # # Create Box observation space
        # obs_shape = flattened_obs.shape


        self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(dtype=np.float32, shape=(image_space), low=-np.inf, high=np.inf),
                'state': gym.spaces.Box(dtype=np.float32, shape= (state_space,), low=-np.inf, high=np.inf),
                'reward': gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
                'is_first': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'is_last': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'is_terminal': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'log/reward': gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
            })
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=obs_shape,
        #     dtype=np.float32
        # )
        
    def _process_observation(self, obs):
        """Convert IsaacLab observation to Dreamer format."""
        # if isinstance(obs, dict):
        #     if self.obs_keys:
        #         # Extract specified keys
        #         obs_parts = []
        #         for key in self.obs_keys:
        #             if key in obs:
        #                 obs_part = obs[key]
        #                 if isinstance(obs_part, torch.Tensor):
        #                     obs_part = obs_part.cpu().numpy()
        #                 obs_parts.append(obs_part.flatten())
                
        #         if obs_parts:
        #             processed_obs = np.concatenate(obs_parts)
        #         else:
        #             # Fallback
        #             obs_parts = []
        #             for value in obs.values():
        #                 if isinstance(value, torch.Tensor):
        #                     value = value.cpu().numpy()
        #                 if isinstance(value, np.ndarray):
        #                     obs_parts.append(value.flatten())
        #             processed_obs = np.concatenate(obs_parts) if obs_parts else np.array([0.0])
        #     else:
        #         # Use all values
        #         obs_parts = []
        #         for value in obs.values():
        #             if isinstance(value, torch.Tensor):
        #                 value = value.cpu().numpy()
        #             if isinstance(value, np.ndarray):
        #                 obs_parts.append(value.flatten())
        #         processed_obs = np.concatenate(obs_parts) if obs_parts else np.array([0.0])
        # else:
        #     # Direct observation
        #     if isinstance(obs, torch.Tensor):
        #         obs = obs.cpu().numpy()
        #     processed_obs = obs.flatten()

        processed_obs = [
            {k: to_np(v) for k, v in zip(obs.keys(), values)}
             for values in zip(*obs.values())
        ]

        if len(processed_obs) == 1:
            processed_obs = processed_obs[0]

        return processed_obs
    
    def reset(self, **kwargs):
        """Reset environment and return Dreamer-compatible observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Handle case where IsaacLab returns multiple environment observations
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            obs = obs[0]  # Take first environment's observation
        elif isinstance(obs, dict) and any(isinstance(v, (list, tuple)) for v in obs.values()):
            # Handle dict with multiple env observations
            new_obs = {}
            for k, v in obs.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    new_obs[k] = v[0]
                else:
                    new_obs[k] = v
            obs = new_obs
        
        processed_obs = self._process_observation(obs)
        self._episode_step = 0
        
        # Create Dreamer-compatible info
        dreamer_info = {'is_first': True, 'is_last': False, 'is_terminal': False}
        if isinstance(info, dict):
            dreamer_info.update(info)

        
        return processed_obs, dreamer_info
    
    def step(self, action):
        """Step environment and return Dreamer-compatible results."""
        # Convert action if needed
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Handle multiple environments case
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            obs = obs[0]
            if isinstance(reward, (list, tuple)):
                reward = reward[0]
            if isinstance(terminated, (list, tuple)):
                terminated = terminated[0]
            if isinstance(truncated, (list, tuple)):
                truncated = truncated[0]
        elif isinstance(obs, dict) and any(isinstance(v, (list, tuple)) for v in obs.values()):
            # Handle dict with multiple env observations
            new_obs = {}
            for k, v in obs.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    new_obs[k] = v[0]
                else:
                    new_obs[k] = v
            obs = new_obs
            
            # Also handle reward, terminated, truncated if they're lists
            if isinstance(reward, (list, tuple)):
                reward = reward[0]
            if isinstance(terminated, (list, tuple)):
                terminated = terminated[0]
            if isinstance(truncated, (list, tuple)):
                truncated = truncated[0]
        
        processed_obs = self._process_observation(obs)
        self._episode_step += 1
        
        # Convert reward to float
        if isinstance(reward, torch.Tensor):
            reward = float(reward.item())
        else:
            reward = float(reward)
        
        # Handle done flag (Dreamer uses single done flag)
        done = terminated or truncated
        
        # Create Dreamer-compatible info
        dreamer_info = {
            'is_first': False,
            'is_last': done,
            'is_terminal': terminated,
        }
        if isinstance(info, dict):
            dreamer_info.update(info)
        
        if isinstance(reward, torch.Tensor):
            reward = to_np(reward)
        if done.numel() == 1:
            done = done.item()
        else:
            done = to_np(done)
        return processed_obs, reward, done, dreamer_info


class IsaacLabMultiEnvWrapper:
    """
    Wrapper for handling multiple IsaacLab environments in parallel.
    This is useful when IsaacLab creates multiple environment instances.
    """
    
    def __init__(self, env, num_envs=None, obs_keys=None):
        """
        Args:
            env: IsaacLab environment
            num_envs: Number of parallel environments (if None, auto-detect)
            obs_keys: Observation keys to extract
        """
        self.env = env
        self.obs_keys = obs_keys
        
        # Detect number of environments
        if num_envs is None:
            sample_obs, _ = env.reset()
            if isinstance(sample_obs, dict):
                # Check if any values are lists/arrays with multiple entries
                for v in sample_obs.values():
                    if hasattr(v, 'shape') and len(v.shape) > 1:
                        num_envs = v.shape[0]
                        break
            elif hasattr(sample_obs, 'shape') and len(sample_obs.shape) > 1:
                num_envs = sample_obs.shape[0]
            else:
                num_envs = 1
        
        self.num_envs = num_envs
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        self.action_space = self.env.action_space
        
        # Create observation space based on single environment
        sample_obs, _ = self.env.reset()
        # single_obs = self._extract_single_env_obs(sample_obs, 0)

        obs_space = sample_obs["state"].shape[1]
        image_space = sample_obs["image"][0].shape
        
        # Process through IsaacLab wrapper to get proper shape
        temp_wrapper = IsaacLabToDreamerWrapper(self.env, self.obs_keys)
        processed_obs = temp_wrapper._process_observation(sample_obs)


        
        self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(dtype=np.float32, shape=(image_space), low=-np.inf, high=np.inf),
                'state': gym.spaces.Box(dtype=np.float32, shape= (obs_space,), low=-np.inf, high=np.inf),
                'reward': gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
                'is_first': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'is_last': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'is_terminal': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'log/reward': gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
            })
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=processed_obs.shape,
        #     dtype=np.float32
        # )
    
    # def _extract_single_env_obs(self, obs, env_idx):
    #     """Extract observation for a single environment."""
    #     # if isinstance(obs, dict):
    #     #     single_obs = {}
    #     #     for k, v in obs.items():
    #     #         if hasattr(v, 'shape') and len(v.shape) > 1 and v.shape[0] == self.num_envs:
    #     #             single_obs[k] = v[env_idx]
    #     #         else:
    #     #             single_obs[k] = v
    #     # elif hasattr(obs, 'shape') and len(obs.shape) > 1 and obs.shape[0] == self.num_envs:
    #     #     single_obs = obs[env_idx]
    #     # else:
    #     #     single_obs = obs
    #     single_obs = {k: v[env_idx] for k, v in obs.items()}
            
    #     return single_obs
    
    def create_single_env_wrapper(self, env_idx):
        """Create a wrapper for a single environment index."""
        return SingleEnvFromMulti(self, env_idx)


class SingleEnvFromMulti:
    """
    Creates a single environment interface from a multi-environment IsaacLab setup.
    """
    
    def __init__(self, multi_env_wrapper, env_idx):
        self.multi_env = multi_env_wrapper
        self.env_idx = env_idx
        self.action_space = multi_env_wrapper.action_space
        self.observation_space = multi_env_wrapper.observation_space
        
        # Create the actual wrapper
        self.wrapper = IsaacLabToDreamerWrapper(multi_env_wrapper.env, multi_env_wrapper.obs_keys)
        
    def reset(self, **kwargs):
        """Reset the specific environment."""
        obs, info = self.multi_env.env.reset(**kwargs)
        # single_obs = self.multi_env._extract_single_env_obs(obs, self.env_idx)
        processed_obs = self.wrapper._process_observation(obs)
        
        
        # if isinstance(info, dict):
        #     dreamer_info.update(info)
        
        # processed_obs.update(dreamer_info)
        return processed_obs #, info
    
    def step(self, action):
        """Step the specific environment."""
        # For multi-env, we need to handle this differently
        # This is a simplified version - you might need to adapt based on your IsaacLab setup
        # full_action = np.zeros((self.multi_env.num_envs,) + action.shape)
        # full_action[self.env_idx] = action[self.env_idx]
        
        obs, reward, terminated, truncated, info = self.multi_env.env.step(action)
        
        # Extract results for this specific environment
        # single_obs = self.multi_env._extract_single_env_obs(obs, self.env_idx)
        processed_obs = self.wrapper._process_observation(obs)
        
        # single_reward = reward[self.env_idx] if hasattr(reward, '__getitem__') else reward
        # single_terminated = terminated[self.env_idx] if hasattr(terminated, '__getitem__') else terminated
        # single_truncated = truncated[self.env_idx] if hasattr(truncated, '__getitem__') else truncated
        
        # if isinstance(single_reward, torch.Tensor):
        #     single_reward = float(single_reward.item())
        
        done = terminated | truncated
        
        dreamer_info = {
            'is_first': False,
            'is_last': done,
            'is_terminal': terminated,
        }
        if isinstance(info, dict):
            dreamer_info.update(info)
        
        return processed_obs, to_np(reward), to_np(done), dreamer_info
    
    def close(self):
        """Close environment."""
        pass  # The main environment will be closed by the multi_env wrapper