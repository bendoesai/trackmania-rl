import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import ReplayBuffer
import networks



DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def build_model(model, obs_space, hidden, act_space) -> nn.Module:
    model_map = {
        'basicnet': networks.basicnet,
    }

    if model.lower() not in model_map:
        raise ValueError(f'Model {model} not recognized. Choose from {list(model_map.keys())}')
    
    return model_map[model.lower()](obs_space, hidden, act_space)


class DummyAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'DummyAgent'
    
    def act(self, obs):
        """
        simplistic policy for LIDAR observations
        """
        return np.array([1.0, 0.0, 0.0])



class VPGAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'vpg'
        self.act_space = act_space
        self.gamma = config['gamma']
        
        # Storage
        self.saved_log_probs = []
        self.rewards = []
        self.actions = []
        
        # Hook for external policy network
        self.policy = build_model(config['actor_model'], obs_space, config['hidden'], act_space * 2)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
            
        action_params = self.policy(obs)
        
        # Split into means and log stds
        action_dim = action_params.shape[-1] // 2
        means = action_params[..., :action_dim]
        log_stds = action_params[..., action_dim:]
        
        # Clamp log_stds for stability
        stds = log_stds.exp()
        
        return means, stds
    
    def act(self, obs, eval = False):
        means, stds = self.forward(obs)
        if not eval:
            stds = torch.clamp(stds, min=0.1, max=8)
        dist = Normal(means, stds)
        raw_action = dist.rsample()  # Use reparameterization trick
        action = torch.tanh(raw_action)  # Squash action to (-1,1)

        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)  # Apply tanh correction

        if not eval:
            self.saved_log_probs.append(log_prob)
            self.actions.append(action)

        return action.detach().numpy()
    
    def update(self):
        returns = self._compute_returns()
        policy_loss = -torch.stack(self.saved_log_probs) * returns
        policy_loss = policy_loss.mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        for param in self.policy.parameters():
            logger.debug(param.grad.norm().item() if param.grad is not None else "No grad")
        self.optimizer.step()
        
        self.saved_log_probs = []
        self.rewards = []
        self.actions = []
        
        return policy_loss.item()
    
    def _compute_returns(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns



class TRPOAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'trpo'
        self.act_space = act_space[0]
        self.gamma = config['gamma']
        self.max_kl = config['max_kl']
        
        # Storage
        self.saved_log_probs = []
        self.rewards = []
        self.actions = []
        
        # Hook for external policy network
        self.policy = None
        self.value = None
    
    def forward(self, obs):
        if self.policy is None:
            raise ValueError("Policy network not set")
        return self.policy(obs)
    
    def act(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            means, stds = self.forward(obs)
            dist = Normal(means, stds)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
        self.saved_log_probs.append(log_prob)
        self.actions.append(action)
        return action.numpy()
    
    def update(self):
        # Compute advantages
        returns = self._compute_returns()
        advantages = returns
        if self.value is not None:
            states = torch.stack([s for s in self.states])
            values = self.value(states).detach()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # TRPO update
        policy_loss = self._trpo_step(advantages)
        
        self.saved_log_probs = []
        self.rewards = []
        self.actions = []
        
        return policy_loss
    
    def _trpo_step(self, advantages):
        # Simplified TRPO update
        # In practice, you'd want to implement the full trust region update
        policy_loss = -(torch.stack(self.saved_log_probs) * advantages).mean()
        return policy_loss.item()



class PPOAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'ppo'
        self.act_space = act_space[0]
        self.gamma = config['gamma']
        self.clip_ratio = config['clip_ratio']
        self.vf_coef = config['vf_coef']
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Hook for external networks
        self.policy = None
        self.value = None
    
    def forward(self, obs):
        if self.policy is None:
            raise ValueError("Policy network not set")
        return self.policy(obs)
    
    def act(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            means, stds = self.forward(obs)
            dist = Normal(means, stds)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            value = self.value(obs) if self.value is not None else torch.zeros(1)
            
        self.states.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action.numpy()
    
    def update(self):
        returns = self._compute_returns()
        advantages = returns
        if self.value is not None:
            advantages = returns - torch.stack(self.values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_loss, value_loss = self._ppo_step(advantages, returns)
        total_loss = policy_loss + self.vf_coef * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        return total_loss.item()
    
    def _ppo_step(self, advantages, returns):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        
        # Policy loss
        means, stds = self.forward(states)
        dist = Normal(means, stds)
        new_log_probs = dist.log_prob(actions).sum(-1)
        ratio = (new_log_probs - old_log_probs).exp()
        
        policy_loss1 = ratio * advantages
        policy_loss2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        # Value loss
        value_loss = 0
        if self.value is not None:
            values = self.value(states)
            value_loss = F.mse_loss(values, returns)
        
        return policy_loss, value_loss



class DDPGAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'ddpg'
        self.act_space = act_space[0]
        self.gamma = config['gamma']
        self.tau = config['tau']
        
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # Hooks for external networks
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None
    
    def act(self, obs, explore=True):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            action = self.actor(obs)
            if explore:
                noise = torch.randn_like(action) * 0.1
                action += noise
            action = torch.clamp(action, -1, 1)
        return action.numpy()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute critic loss
        Q_targets_next = self.target_critic(next_states, self.target_actor(next_states))
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )



class TD3Agent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'td3'
        self.act_space = act_space[0]
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.policy_delay = config['policy_delay']
        
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.total_it = 0
        
        # Hooks for external networks
        self.actor = None
        self.critic1 = None
        self.critic2 = None
        self.target_actor = None
        self.target_critic1 = None
        self.target_critic2 = None
    
    def act(self, obs, explore=True):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            action = self.actor(obs)
            if explore:
                noise = torch.randn_like(action) * 0.1
                noise = torch.clamp(noise, -0.5, 0.5)
                action += noise
            action = torch.clamp(action, -1, 1)
        return action.numpy()
    
    def update(self):
        self.total_it += 1
        
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Select next actions with noise
        noise = torch.randn_like(actions) * 0.2
        noise = torch.clamp(noise, -0.5, 0.5)
        next_actions = self.target_actor(next_states) + noise
        next_actions = torch.clamp(next_actions, -1, 1)
        
        # Compute critic loss
        Q1_next = self.target_critic1(next_states, next_actions)
        Q2_next = self.target_critic2(next_states, next_actions)
        Q_next = torch.min(Q1_next, Q2_next)
        Q_target = rewards + (self.gamma * Q_next * (1 - dones))
        
        critic1_loss = F.mse_loss(self.critic1(states, actions), Q_target)
        critic2_loss = F.mse_loss(self.critic2(states, actions), Q_target)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = 0
        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )



class SACAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'sac'
        self.act_space = act_space[0]
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # Hooks for external networks
        self.actor = None
        self.critic1 = None
        self.critic2 = None
        self.target_critic1 = None
        self.target_critic2 = None
        
        # Optional automatic entropy tuning
        self.target_entropy = -np.prod(act_space)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = torch.exp(self.log_alpha)
    
    def act(self, obs, explore=True):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            mean, log_std = self.actor(obs)
            if explore:
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.rsample()  # reparameterization trick
            else:
                action = mean
            action = torch.tanh(action)  # squash to [-1, 1]
        return action.numpy()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update critics
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mean, next_std)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions).sum(-1, keepdim=True)
            next_actions = torch.tanh(next_actions)
            
            # Target Q-values
            Q1_next = self.target_critic1(next_states, next_actions)
            Q2_next = self.target_critic2(next_states, next_actions)
            Q_next = torch.min(Q1_next, Q2_next)
            Q_target = rewards + (self.gamma * (1 - dones) * (Q_next - self.alpha * next_log_probs))
        
        # Current Q-values
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_Q1, Q_target)
        critic2_loss = F.mse_loss(current_Q2, Q_target)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        actions_pred = dist.rsample()
        log_probs = dist.log_prob(actions_pred).sum(-1, keepdim=True)
        actions_pred = torch.tanh(actions_pred)
        
        Q1_pred = self.critic1(states, actions_pred)
        Q2_pred = self.critic2(states, actions_pred)
        Q_pred = torch.min(Q1_pred, Q2_pred)
        
        actor_loss = (self.alpha * log_probs - Q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = torch.exp(self.log_alpha)
        
        # Update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )