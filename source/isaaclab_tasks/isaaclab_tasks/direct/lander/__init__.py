# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Planetary Lander environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-PlanetaryLander-Direct-v0",
    entry_point=f"{__name__}.lander_env:LanderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lander_env:LanderEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_cfg.yaml",
    },
)

gym.register(
    id="Isaac-PlanetaryLander-Direct-States-v0",
    entry_point=f"{__name__}.lander_states_env:LanderStatesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lander_states_env:LanderStatesEnvCfg",
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_states_cfg.yaml",
    },
)

gym.register(
    id="Isaac-PlanetaryLander-Direct-6DOF-v0",
    entry_point=f"{__name__}.lander_6dof_env:Lander6DOFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lander_6dof_env:Lander6DOFEnvCfg",
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_6dof_cfg.yaml",
    },
)

gym.register(
    id="Isaac-PlanetaryLander-Direct-6DOF-Manual-v0",
    entry_point=f"{__name__}.lander_6dof_manual_env:Lander6DOFManualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lander_6dof_manual_env:Lander6DOFManualEnvCfg",
        "dreamer_cfg_entry_point": f"{agents.__name__}:dreamer_6dof_manual_cfg.yaml",
    },
)