import gymnasium as gym
from gymnasium import spaces
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleEnv(gym.Env):
    """
    One action, zero observation, one timestep long, +1 reward every timestep.
    Modified for continuous action space.
    """
    
    def __init__(self):
        # Continuous action space: 1-dimensional bounded to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: empty but with proper shape for compatibility
        self.observation_space = spaces.Box(low=np.array([]), high=np.array([]), 
                                           shape=(0,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Return empty observation and info
        return np.array([], dtype=np.float32), {}
        
    def step(self, action):
        # Always return +1 reward regardless of action
        reward = 1.0
        # Episode is always done after one step
        terminated = True
        # No truncation
        truncated = False
        # Return empty observation
        observation = np.array([], dtype=np.float32)
        # Empty info dict
        info = {}
        
        return observation, reward, terminated, truncated, info


class RandomObsRewardEnv(gym.Env):
    """
    One action, random +1/-1 observation, one timestep long, 
    obs-dependent +1/-1 reward every time.
    Modified for continuous action space.
    """
    
    def __init__(self):
        # Continuous action space: 1-dimensional bounded to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: single value, either +1 or -1
        self.observation_space = spaces.Box(low=-1, high=1, 
                                           shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate random observation (+1 or -1)
        self.obs = np.array([2 * self.np_random.integers(0, 2) - 1], dtype=np.float32)
        return self.obs, {}
        
    def step(self, action):
        # Reward equals the observation value
        reward = float(self.obs[0])
        # Episode is always done after one step
        terminated = True
        truncated = False
        # Return info dict
        info = {}
        
        return self.obs, reward, terminated, truncated, info


class TwoStepDelayedRewardEnv(gym.Env):
    """
    One action, zero-then-one observation, two timesteps long, 
    +1 reward at the end.
    Modified for continuous action space.
    """
    
    def __init__(self):
        # Continuous action space: 1-dimensional bounded to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: single value, either 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, 
                                           shape=(1,), dtype=np.float32)
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # First observation is 0
        observation = np.array([0], dtype=np.float32)
        return observation, {}
        
    def step(self, action):
        self.step_count += 1
        
        if self.step_count == 1:
            # First step: zero reward, not done
            reward = 0.0
            terminated = False
            # Observation becomes 1
            observation = np.array([1], dtype=np.float32)
        else:
            # Second step: +1 reward, done
            reward = 1.0
            terminated = True
            # Keep observation as 1
            observation = np.array([1], dtype=np.float32)
            
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info


class ActionDependentRewardEnv(gym.Env):
    """
    Two actions, zero observation, one timestep long, 
    action-dependent +1/-1 reward.
    Modified for continuous action space.
    """
    
    def __init__(self):
        # Continuous action space: 1-dimensional bounded to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: empty but with proper shape
        self.observation_space = spaces.Box(low=np.array([]), high=np.array([]), 
                                           shape=(0,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Return empty observation
        return np.array([], dtype=np.float32), {}
        
    def step(self, action):
        # Reward depends on action sign: positive actions give +1, negative give -1
        reward = 1.0 if action[0] >= 0 else -1.0
        
        # Episode is always done after one step
        terminated = True
        truncated = False
        # Return empty observation
        observation = np.array([], dtype=np.float32)
        info = {}
        
        return observation, reward, terminated, truncated, info


class ActionObsDependentRewardEnv(gym.Env):
    """
    Two actions, random +1/-1 observation, one timestep long,
    action-and-obs dependent +1/-1 reward.
    Modified for continuous action space.
    """
    
    def __init__(self):
        # Continuous action space: 1-dimensional bounded to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: single value, either +1 or -1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate random observation (+1 or -1)
        self.obs = np.array([2 * self.np_random.integers(0, 2) - 1], dtype=np.float32)
        return self.obs, {}
        
    def step(self, action):
        # Reward is +1 if action sign matches observation sign, -1 otherwise
        obs_positive = self.obs[0] > 0
        action_positive = action[0] >= 0
        reward = 1.0 if obs_positive == action_positive else -1.0
        
        # Episode is always done after one step
        terminated = True
        truncated = False
        info = {}
        
        return self.obs, reward, terminated, truncated, info