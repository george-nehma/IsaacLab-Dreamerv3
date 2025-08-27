import random
from typing import Dict

import gym
import gym.spaces
import numpy as np
from gym import Env
from gym import spaces
import quaternion



class LanderVehicle:
    def __init__(self, mass=500, inertia = 10, thruster_directions=None, thruster_positions=None, Isp=225):
        self.mass = np.array([mass])
        self.inertia_yy = self.mass*(5**2)

def moon_env():
    radius = 1737400 # [m]
    mass = 7.34767e22 # [kg]
    return radius, mass

def mars_env():
    radius = 3369200 # [m]
    mass = 6.4169e23  # [kg]
    return radius, mass

def asteroid_env(): # Cleopatra
    radius = 476000 # [m]
    mass = 9.39e20  # [kg]
    return radius, mass

PLANET_SAMPLERS = {
    'Moon': moon_env,
    'Mars': mars_env,
    'Asteroid': asteroid_env,
}


class LanderEnv(Env):
    """
    Simulates a lander vehicle to land from a range of various initial conditions at any given location

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, name, planet=None, lander_veh=LanderVehicle(), eval_mode=False, num_episodes=5, max_steps=50000):
        del name
        if callable(planet):
            self.radius, self.planet_mass = planet
        elif isinstance(planet, str):
            self.radius, self.planet_mass = PLANET_SAMPLERS[planet]
        elif planet is None:
            self.radius, self.planet_mass = moon_env()
        else:
            raise NotImplementedError(planet)
        
        self._state = None
        self.num_episodes = num_episodes
        self.lander = lander_veh
        self.gravity = np.array([0,-6.6743e-11*self.planet_mass/(self.radius**2)])
        self.forceBody = None

        self.rlim = np.array([2,2])
        self.vlim = np.array([0.5,0.5])

        self.max_steps = max_steps
        self.step_count = 0

        # State space: [x,z, x_vel, z_vel]
        self.stateHigh = np.array([
            np.inf,   # max x position [m]
            np.inf,   # max z position [m]
            np.inf,   # max x velocity [m]
            np.inf,   # max z velocity [m]
            np.pi/2,  # max roll angle [rad]
            np.inf    # max angular vel [rad/s]
        ], dtype=np.float32)

        self.stateLow = np.array([
            -np.inf,   # min x position [m]
            0,        # min z position [m]
            -np.inf,   # min x velocity [m]
            -np.inf,   # min z velocity [m]
            -np.pi/2, # min roll angle [rad]
            -np.inf    # min angular vel [rad/s]
        ], dtype=np.float32)

        # Action space: [+x thrust, -x thrust, vertical thrust]
        # 4 RCS thrusters per side 
        self.actionHigh = np.array([3000, 3000, 3000, 8000], dtype=np.float32) # max thrust of RCS thrusters [N] and moment 

        self.actionLow = np.array([0, 0, 0, -8000], dtype=np.float32) # min thrust of RCS thrusters [N] and moment

        self.obs_space = gym.spaces.Dict({
                'state': gym.spaces.Box(dtype=np.float32, shape= self.stateHigh.shape, low=self.stateLow, high=self.stateHigh),
                'reward': gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
                'is_first': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'is_last': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'is_terminal': gym.spaces.Box(dtype=bool, shape=(), low=0, high=1),
                'log/reward': gym.spaces.Box(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
            })
        # we convert the actions from [-1, 1] to [-0.1, 0.1] in the step() function
        self.act_space = gym.spaces.Box(dtype=np.float32, shape=self.actionHigh.shape ,low=self.actionLow, high=self.actionHigh)
        
        self._np_random = np.random.RandomState()
        self._seed = None

        self.prev_shaping = None
        self.spower = False
        self.mpower = False
        self.landed = False
        self.crashed = False
        self.missed = False

        self.reset()

    @property
    def observation_space(self):
        return self.obs_space
    
    @property
    def action_space(self):
        return self.act_space

    def seed(self, seed=None):
        self._seed = seed
        self._np_random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset_model(self):
        self._vel = np.array([np.random.uniform(-20,-10), 
                              np.random.uniform(-30,-20)]).T # initial velocity [m/s]
        
        self._v0 = self._vel
        
        self._position = np.array([np.random.uniform(0,200), 
                                   np.random.uniform(200,300)]).T # initial position [m]
        
        self._angle = np.array([np.random.uniform(-np.pi/6, np.pi/6)]) # initial roll angle [rad]

        self._angvel = np.array([np.random.uniform(-0.3, 0.3)]) # initial angular velocity [rad/s]
        
        self._state = np.hstack((self._position, self._vel, self._angle, self._angvel))

        self.step_count = 0
        self.done = False

        self._prev_action = np.zeros_like(self.action_space.low)

        return self._obs(is_first=True)

    def render(self, mode="human"):
        # Optional: Add Matplotlib rendering if needed
        pass

    def reset(self):
        obs = self.reset_model()
        self.done = False
        return obs

    
    def _obs(self, reward=0.0, is_first=False, is_last=False, is_terminal=False):

        # Ensure state is finite and clipped to obs space
        state = np.clip(self._state, self.obs_space["state"].low, self.obs_space["state"].high)
        if not np.isfinite(state).all():
            is_last = True
            is_terminal = True

        if is_last or is_terminal:
            self.step_count = 0

        obs = {
            "state": state.astype(np.float32),
            "reward": float(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            'log/reward': float(reward),
            'log/dynstates': state.astype(np.float32)
        }
        return obs
    
    # runge-kutta fourth-order numerical integration
    def rk4(self, func, tk, yk, dt, *args):
        """
        single-step fourth-order numerical integration (RK4) method
        func: system of first order ODEs
        tk: current time step
        _yk: current state vector [y1, y2, y3, ...]
        _dt: discrete time step size
        **kwargs: additional parameters for ODE system
        returns: y evaluated at time k+1
        """

        # evaluate derivative at several stages within time interval
        f1 = func(tk, yk, *args)
        f2 = func(tk + dt / 2, yk + (f1 * (dt / 2)), *args)
        f3 = func(tk + dt / 2, yk + (f2 * (dt / 2)), *args)
        f4 = func(tk + dt, yk + (f3 * dt), *args)

        # return an average of the derivative over tk, tk + dt
        self._state = yk + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)
        self._position = self._state[0:2]
        self._vel = self._state[2:4]
        self._angle = self._state[4]
        self._angvel = self._state[5]

    def sixDOFDynamics(self, t, state, *args):
        action = args[0]
        position = state[0:2]
        vel = state[2:4]
        theta = state[4]
        omega = state[5]

        self.mpower = np.any(action[2])
        self.spower = np.any(action[0:2]) 
        self.tpower = np.any(action[3])
        xthrust = action[0] - action[1]  # total thrust in x direction
        zthrust = action[2]  # total thrust in z direction
        self.forceBody = np.array([xthrust, zthrust])  # force in body frame

        mass = self.lander.mass
        inertia = self.lander.inertia_yy

        moment_body = action[3]
        acc_body = self.forceBody/mass + np.array([-self.gravity[1]*np.sin(theta), self.gravity[1]*np.cos(theta)])

        position_dot = np.array([vel[0]*np.cos(theta) + vel[1]*np.sin(theta), -vel[0]*np.sin(theta) + vel[1]*np.cos(theta)])
        vel_dot = acc_body + omega*np.array([-vel[1], vel[0]])
        theta_dot = omega
        omega_dot = moment_body/inertia

        x_dot = np.hstack ((position_dot, vel_dot,theta_dot,omega_dot))
        return x_dot
    
    def reward_func(self):

        reward = 0

        self.landed = (self._position[1] <= 0 and
            np.all(np.linalg.norm(self._position) < self.rlim) and
            np.all(np.linalg.norm(self._vel) < self.vlim) and
            np.abs(self._angle) < 0.087 and 
            np.abs(self._angvel) < 0.03)   

        self.crashed = (self._position[1] <= 0 and
            np.all(np.linalg.norm(self._position) > self.rlim) and
            np.all(np.linalg.norm(self._vel) > self.vlim) and
            np.abs(self._angle) > 0.087 and 
            np.abs(self._angvel) > 0.03) 
            
        self.missed = (self._position[1] <= 0 and
            np.all(np.linalg.norm(self._position) > self.rlim) and
            np.all(np.linalg.norm(self._vel) < self.vlim) and
            np.abs(self._angle) < 0.087 and 
            np.abs(self._angvel) < 0.03)

        alpha = -50
        beta = -50
        gamma = -0.03
        delta = -0.3

        reward = alpha * np.linalg.norm(self._position) + beta * np.linalg.norm(self._vel) - 500 * np.abs(self._angle) - 100*np.abs(self._angvel) # - 0 * np.linalg.norm(self._current_action - self._prev_action)**2

        # if self.prev_shaping is not None:
        #     reward = shaping - self.prev_shaping
        # self.prev_shaping = shaping

        reward += gamma * self.spower + delta * self.mpower - 0.5 * self.tpower

        if self.landed:
            reward += 1000
            print("Landed!")
        elif self.crashed:
            reward += -3000
            print("Crashed!")
        elif self.missed:
            reward += 50
            print("Missed!")

        return reward

    def step(self, action):
        if self.done:
            # If episode is done, return a terminal observation
            reward = self.reward_func()
            return self._obs(is_first=False, is_last=True, is_terminal=True), reward, self.done, {"discount": 0.0}
        action = np.clip(action, self.act_space.low, self.act_space.high)
        self._current_action = action
        self.rk4(self.sixDOFDynamics, 1, self._state, 0.1, action)
        reward = self.reward_func()
        self._prev_action = action
        self.step_count += 1
        # Check for non-finite or out-of-bounds state
        nonfinite = not np.isfinite(self._state).all()
        out_of_bounds =  self._position[1] < 0 or np.any(self._position > 1e3)
        if out_of_bounds: print('out of bounds!')
        # out_of_bounds = np.any(self._state[:3] > 1e3) or np.any(self._state[:3] < -1e3)
        self.done = self.step_count >= self.max_steps or nonfinite or out_of_bounds or self.landed or self.crashed or self.missed
        obs = self._obs(
            reward=reward,
            is_first=False,
            is_last=self.done,
            is_terminal=self.landed or self.crashed or self.missed,
        )
        # If state is invalid, reset the environment next step
        if nonfinite or out_of_bounds:
            self.done = True
        return obs, reward, self.done, {"discount": 0.0}