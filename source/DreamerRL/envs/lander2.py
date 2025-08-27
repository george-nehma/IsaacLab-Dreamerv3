import random
from typing import Dict

import gym
import gym.spaces
import numpy as np
from gym import Env
from gym import spaces
import quaternion



class LanderVehicle:
    def __init__(self, mass=2000, inertia_tensor=np.eye(3), thruster_directions=None, thruster_positions=None, Isp=225):
        self.mass = np.array([mass])
        self.inertia = 0.25*self.mass/5*inertia_tensor
        self.Isp = Isp

        if thruster_positions is None:
            self.thruster_positions = np.array([     # all positions in meters in body frame and from CG of vehicle
                [0.0, -2.0, -1.0],
                [0.0, 2.0, -1.0],
                [-2.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],               
            ])
        else:
            self.thruster_positions = thruster_positions

        if thruster_directions is None:
            self.thruster_directions = np.array([    # all directions in body frame (FRD)
                [0.0, -1.0, 0.0], 
                [0.0, 1.0, 0.0], 
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]).T
        else:
            self.thruster_directions = thruster_directions


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

    def __init__(self, name, planet=None, lander_veh=LanderVehicle(), eval_mode=False, num_episodes=5, max_steps=2000):
        del name
        if callable(planet):
            self.radius, self.planet_mass = planet
        elif isinstance(planet, str):
            self.radius, self.planet_mass = PLANET_SAMPLERS[planet]
        elif planet is None:
            self.radius, self.planet_mass = moon_env()
        else:
            raise NotImplementedError(planet)
        
        self._state = np.empty(12)
        self._x = None
        self.num_episodes = num_episodes
        self.task_dim = 16
        self.lander = lander_veh
        self.gravity = np.array([[0],[0],[-6.6743e-11*self.planet_mass/(self.radius**2)]])
        self.forceBody = None

        self.qlim = np.array([360,80,80])
        self.qmgn = np.array([360,67,67])
        self.rlim = np.array([5,5,5])
        self.vlim = np.array([2,2,2])
        self.q_radlim = np.array([0.2,0.2,np.inf])
        self.wlim = np.array([0.2,0.2,0.2])
        self.gslim = 79

        self.goal = np.empty(5, dtype=object)
        self.goal[0] = np.array([0,0,0])
        self.goal[1] = quaternion.from_euler_angles(np.array([0,0,0]))
        self.goal[2] = np.array([0,0,0])
        self.goal[3] = 0
        self.goal[4] = 0

        self.max_steps = max_steps
        self.step_count = 0

        # State space: [vx_error, vy_error, vz_error, q0, q1, q2, q3, wx, wy, wz, rz, tgo, left_front_leg_contact, left_rear_leg_contact, right_front_leg_contact, right_rear_leg_contact]
        self.stateHigh = np.array([
            np.inf,  # max vx error [m/s]
            np.inf,  # max vy error [m/s]
            np.inf,  # max vz error [m/s]
            1.0,     # max q0
            1.0,     # max q1 
            1.0,     # max q2 
            1.0,     # max q3
            20.0,  # max x angular velocity [rad/s]
            20.0,  # max y angular velocity [rad/s]
            20.0,  # max z angular velocity [rad/s]
            np.inf, # z position [m]
            np.inf, # time to go
            #1.0,  # leg contact front left (0.0 or 1.0)
            #1.0,  # leg contact rear left
            #1.0,   # leg contact front right
            #1.0   # leg contact rear right
        ], dtype=np.float32)

        # Action space: [mainEngine, RCS1, RCS2, RCS3, RCS4, RCS5, RCS6, RCS7, RCS8, RCS9, RCS10, RCS11, RCS12, RCS13, RCS14, RCS15, RCS16]
        # 4 RCS thrusters per side 
        self.actionHigh = np.full(4, 5000, dtype=np.float32) # max thrust of RCS thrusters [N] 5000

        self.actionLow = np.full(4, 0, dtype=np.float32) # min thrust of RCS thrusters [N] 1000

        self.obs_space = gym.spaces.Dict({
                'state': gym.spaces.Box(dtype=np.float32, shape= self.stateHigh.shape, low=-self.stateHigh, high=self.stateHigh),
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
        self._goalState = np.array([])
        self._goalLat = np.deg2rad(0.6742)
        self._goalLon = np.deg2rad(23.4371)
        self._betax = np.deg2rad(np.random.uniform(-22.5,22.5)) # roll
        self._betay = np.deg2rad(np.random.uniform(22.5,67.5)) # pitch
        self._betaz = np.deg2rad(np.random.uniform(-22.5,22.5)) # yaw
        self._quatI2E = quaternion.from_euler_angles(np.array([0,-self._goalLat-np.pi/2,self._goalLon])) # rotate to lon and then left hand rotation to lat (-90-lat) Interial to NED frame
        self._quatE2I = self._quatI2E.inverse() # from NED frame to inertial frame

        self._quatB2E = quaternion.from_euler_angles(np.array([self._betax,self._betay,self._betaz])) # quaternion of initial attitude of lander in NED frame
        self._quatB2E = self._quatB2E/self._quatB2E.norm()
        self._vel = np.array([np.random.uniform(-70,-10), 
                              np.random.uniform(-30,30), 
                              np.random.uniform(-90,-70)]).T # initial velocity in NED frame [m/s]
        
        self._v0 = self._vel
        
        self._position = np.array([np.random.uniform(0,2000), 
                                   np.random.uniform(-1000,1000), 
                                   np.random.uniform(2300,2400)]).T # initial position in NED frame [m]
        
        self._angvel = np.array([np.deg2rad(np.random.uniform(-0.0,0.0)),
                                 np.deg2rad(np.random.uniform(-0.6,0.6)), 
                                 np.deg2rad(np.random.uniform(-0.6,0.6))]).T # initial velocity in NED frame [rad/s]
        
        # self._vtarget = np.array([0,0,-2]).T # initial target velocity (t>15sec) [m/s]
        
        self._x = np.empty(5, dtype=object)
        self._x[0] = self._quatB2E
        self._x[1] = self._position
        self._x[2] = self._angvel
        self._x[3] = self._vel
        self._x[4] = self.lander.mass

        self.step_count = 0
        self.done = False

        return self._obs(is_first=True)

    def render(self, mode="human"):
        # Optional: Add Matplotlib rendering if needed
        pass

    def reset(self):
        obs = self.reset_model()
        self.done = False
        return obs

    
    def _obs(self, reward=0.0, is_first=False, is_last=False, is_terminal=False):

        if self._position[2] > 15:
            tau = 20
            v_hat = self._vel - np.array([0,0,-2]).T
            r_hat = self._position - np.array([0,0,15]).T
            
        else: 
            tau = 100
            v_hat = self._vel - np.array([0,0,-1]).T
            r_hat = np.array([0,0,self._position[2]]).T

        self._tgo = np.linalg.norm(r_hat)/np.linalg.norm(v_hat)
        self._vtarget = -np.linalg.norm(self._v0)*(r_hat/np.linalg.norm(r_hat))*(1-np.exp(-self._tgo/tau))

        self._verror = self._vel - self._vtarget # error in velocity

        
        self._state[0:3] = self._verror
        self._state[3:7] = quaternion.as_float_array(self._quatB2E/self._quatB2E.norm())
        self._state[7:10] = self._angvel
        self._state[10] = self._position[2]
        self._state[11] = self._tgo

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
            'log/dynstates': np.concatenate([quaternion.as_float_array(self._x[0])] + list(self._x[1:5])).astype(np.float32)
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
        self._x = yk + (dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)
        self._quatB2E = self._x[0]/self._x[0].norm()
        self._position = self._x[1]
        self._angvel = self._x[2]
        self._vel = self._x[3]
        self.lander.mass = self._x[4]
    
    def sixDOFDynamics(self, t, state, *args):
        action = args[0]
        quatB2E = state[0]
        position = state[1]
        angvel = state[2]
        vel = state[3]
        mass = state[4]

        self.lander.inertia = 0.25*mass[0]/5*np.eye(3) # inertia tensor in body frame

        self.forceBody = np.dot(self.lander.thruster_directions, action.reshape(-1,1))
        momentsBody = np.sum(np.cross(self.lander.thruster_positions,self.forceBody.flatten()),axis=0)

        _quatx = np.array([[0, -quatB2E.z, quatB2E.y], 
                                [quatB2E.z, 0, -quatB2E.x],
                                [-quatB2E.y, quatB2E.x, 0]])
        
        Xi = np.vstack((quatB2E.w*np.eye(3) + _quatx, [[-quatB2E.x, -quatB2E.y, -quatB2E.z]]))
        Psi = np.vstack((quatB2E.w*np.eye(3) - _quatx, [[-quatB2E.x, -quatB2E.y, -quatB2E.z]]))

        A_N2B = Xi.T @ Psi
        forceInertial = A_N2B.T @ self.forceBody

        _angvelx = np.array([[0, -angvel[2], angvel[1]], 
                                [angvel[2], 0, -angvel[0]],
                                [-angvel[1], angvel[0], 0]])

        angvel_dot = np.linalg.inv(self.lander.inertia) @ (-_angvelx @ self.lander.inertia @ angvel + momentsBody)

        quat_dot = 0.5*(Xi@angvel)
        quat_dot = quaternion.as_quat_array(quat_dot)

        position_dot = vel
        vel_dot = forceInertial/mass + self.gravity

        mass_dot = -np.sum(self.forceBody)/(self.lander.Isp * 9.81)

        x_dot = np.empty(5, dtype=object)
        x_dot[0] = quat_dot
        x_dot[1] = position_dot
        x_dot[2] = angvel_dot
        x_dot[3] = vel_dot.flatten()
        x_dot[4] = mass_dot
        return x_dot
    
    def reward_func(self):
        euler = quaternion.as_euler_angles(self._quatB2E)
        q_lims = np.any(euler > self.qlim)
        approach_lim = np.sum(-np.maximum(0,euler - self.qmgn))

        gs = np.arctan(np.sqrt(self._position[1]**2 + self._position[2]**2)/self._position[0])

        self.landed = (self._position[2] < 0 and
            np.all(np.linalg.norm(self._position) < self.rlim) and
            np.all(np.linalg.norm(self._vel) < self.vlim) and
            np.all(euler < self.q_radlim) and
            np.all(self._angvel < self.wlim and gs < self.gslim))
        
        if self.landed:
            k = 10
        else: k = 0

        alpha = -0.01
        beta = -0.05
        gamma = -100
        delta = -20
        eta = 0.01

        reward = alpha * np.linalg.norm(self._verror) + beta * np.linalg.norm(self.forceBody) +  gamma * q_lims + delta * approach_lim + eta + k

        return reward

    def step(self, action):
        if self.done:
            # If episode is done, return a terminal observation
            reward = self.reward_func()
            return self._obs(is_first=False, is_last=True, is_terminal=True), reward, self.done, {"discount": 0.0}
        action = np.clip(action, self.act_space.low, self.act_space.high)
        self.rk4(self.sixDOFDynamics, 1, self._x, 0.2, action)
        reward = self.reward_func()
        self.step_count += 1
        # Check for non-finite or out-of-bounds state
        nonfinite = not np.isfinite(self._state).all()
        out_of_bounds =  np.any(self._position[2] < 0) or np.any(self._position > 1e4)
        # out_of_bounds = np.any(self._state[:3] > 1e3) or np.any(self._state[:3] < -1e3)
        self.done = self.step_count >= self.max_steps or nonfinite or out_of_bounds or self.landed
        obs = self._obs(
            reward=reward,
            is_first=False,
            is_last=self.done,
            is_terminal=self.landed,
        )
        # If state is invalid, reset the environment next step
        if nonfinite or out_of_bounds:
            self.done = True
        return obs, reward, self.done, {"discount": 0.0}