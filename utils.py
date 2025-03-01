import torch
from torch.distributions import Normal
import numpy as np
import random
from collections import deque, namedtuple
from copy import copy

def flatten_and_norm(obs):
    """Flattens a tuple of tuples with varying lengths into a single tuple."""
    flat = np.concatenate([np.ravel(arr) for arr in obs])
    return flat

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(self.device), 
                torch.FloatTensor(np.array(actions)).to(self.device), 
                torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(self.device), 
                torch.FloatTensor(np.array(next_states)).to(self.device), 
                torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(self.device))
    
    def __len__(self):
        return len(self.buffer)
    
class PriorityBuffer:
    # replaybuffer but sorted and sampled from front. sort every time a done is found
    pass

class GaussianNoise:
    '''standard gaussian noise'''

    def __init__(self, act_space, up_lim, down_lim):
        
        self.mu = (up_lim + down_lim) / 2
        self.std = (up_lim - down_lim) / 4
        
        self.mu = self.mu * torch.ones(act_space)
        self.std = self.std * torch.ones(act_space)

        self.dist = Normal(self.mu, self.std)

    def reset(self):
        pass

    def sample(self, current_reward):
        return self.dist.rsample()
    
    def update_max_reward(self, current_reward):
        pass


class OUNoise:
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size=1, mu=0, theta=1.0, sigma=0.5, dt=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mu)

    def sample(self, current_reward):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
    def update_max_reward(self, current_reward):
        """Update the maximum reward seen so far."""
        pass

class RewardBasedOUNoise:
    """Ornstein-Uhlenbeck noise that scales based on progress/reward."""

    def __init__(self, size=1, mu=0, theta=0.5, base_sigma=1, dt=0.1, max_reward_seen=0.1):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.base_sigma = base_sigma
        self.dt = dt
        self.max_reward_seen = max_reward_seen  # Initialize with small value
        self.reset()

    def reset(self):
        """Reset the internal state to mean."""
        self.state = copy(self.mu)

    def update_max_reward(self, current_reward):
        """Update the maximum reward seen so far."""
        self.max_reward_seen = max(self.max_reward_seen, current_reward)
        
    def sample(self, current_reward):
        """Generate noise based on reward progress."""
        if self.max_reward_seen <= 0.3:  # Avoid division by zero
            progress_factor = 1.0
        else:
            # Calculate how close we are to frontier (1.0 = at frontier, 0.0 = far from frontier)
            progress_ratio = current_reward / self.max_reward_seen
            # Apply exponential scaling to emphasize frontier exploration
            progress_factor = np.exp(-5 * (1-progress_ratio))
            
        # Scale sigma based on progress
        effective_sigma = self.base_sigma * progress_factor
        
        # Standard OU process with scaled sigma
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + effective_sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state