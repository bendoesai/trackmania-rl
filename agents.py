import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

from utils import ReplayBuffer, GaussianNoise, OUNoise, RewardBasedOUNoise
import networks



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#DEVICE = 'cpu'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()



def build_model(model, obs_space, hidden, act_space) -> nn.Module:
    model_map = {
        'basicnet': networks.basicnet,
    }

    if model.lower() not in model_map:
        raise ValueError(f'Model {model} not recognized. Choose from {list(model_map.keys())}')
    
    return model_map[model.lower()](obs_space, hidden, act_space).to(DEVICE)

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
        
        self.to(DEVICE)

        # Hook for external policy network
        self.policy = build_model(config['actor_model'], obs_space, config['hidden'], act_space * 2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), config['lr'])

    def forward(self, obs): 
        action_params = self.policy(obs)
        
        # Split into means and log stds
        action_dim = action_params.shape[-1] // 2
        means = action_params[..., :action_dim]
        log_stds = action_params[..., action_dim:]
        
        # "Clamp" log_stds for stability
        log_stds = 2*torch.tanh(log_stds/2.0)
        stds = log_stds.exp()
        
        return means, stds
    
    def act(self, obs, eval = False):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(DEVICE)
        
        means, stds = self.forward(obs)

        dist = Normal(means, stds)
        raw_action = dist.rsample()  # Use reparameterization trick
        action = torch.tanh(raw_action)  # Squash action to (-1,1)

        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)  # Apply tanh correction

        if not eval:
            self.saved_log_probs.append(log_prob)
            self.actions.append(action)

        return action.cpu().detach().numpy()
    
    def update(self):
        returns = self._compute_returns()

        policy_loss = -torch.stack(self.saved_log_probs) * returns
        policy_loss = policy_loss.mean()

        print(f"Policy Loss: {policy_loss.item()}")

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
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
        returns = torch.FloatTensor(returns).to(DEVICE)
        # print(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # print(returns)
        return returns



class TRPOAgent(nn.Module):
    '''
    Trust Region Policy Optimization
    - on-policy
    - stochastic
    '''
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        raise NotImplementedError("TPRO not yet implemented")
    
    def forward(self, obs):
        pass
    
    def act(self, obs):
        pass
    
    def store_transition(self, state, action, reward, next_state, done):
        pass

    def update(self):
        # Compute advantages
        pass
    
    def _trpo_step(self, advantages):
        pass



class PPOAgent(nn.Module):
    '''
    Proximal Policy Estimation
    - On Policy
    - Stochastic
    '''
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'ppo'
        self.act_space = act_space[0] if isinstance(act_space, list) else act_space
        
        self.gamma = config['gamma']
        self.clip_ratio = config['clip_ratio']
        self.vf_coef = config['vf_coef']
        self.entropy_coef = config['entropy']
        self.lam = config['lambda']
        self.ppo_epochs = config['ppo_epochs']
        self.max_norm = config['max_grad_norm']

        self.up_bound = config['action_bound_up']
        self.down_bound = config['action_bound_down']

        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Hook for external networks
        self.policy = build_model(config['actor_model'], obs_space, config['hidden'], act_space * 2)
        self.value = build_model(config['critic_model'], obs_space, config['hidden'], 1)

        #TODO: Implement separate policy and value optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': config['actor_lr'], 'eps': 1e-5},
            {'params': self.value.parameters(), 'lr': config['critic_lr'], 'eps': 1e-5},
        ])

        self.to(DEVICE)
    
    def forward(self, obs):
        action_params = self.policy(obs)

        action_dim = action_params.shape[-1] // 2
        means = action_params[..., :action_dim]
        log_stds = action_params[..., action_dim:]

        # "Clamp" log_stds for stability
        log_stds = torch.clamp(log_stds, -20, 2)
        stds = log_stds.exp()
        
        return means, stds
    
    def act(self, obs, eval=False):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(DEVICE)

        with torch.no_grad():
            means, stds = self.forward(obs)
            dist = Normal(means, stds)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1)
            action = torch.clamp(action, min=self.down_bound, max=self.up_bound)
            value = self.value(obs)

            #tanh correction
            #log_prob -= torch.sum(torch.log(torch.clamp(1 - action.pow(2), min=1e-2)), dim=-1)

        if not eval:    
            self.states.append(obs)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
        print(action, log_prob)
        
        return action.detach().cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def _process_rewards(self):
        # Convert stored experiences to tensors
        rewards = torch.FloatTensor(self.rewards).to(DEVICE)
        values = torch.cat(self.values)
        dones = torch.FloatTensor(self.dones).to(DEVICE)
        
        # For GAE calculation
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Loop backwards through rewards
        for t in reversed(range(len(rewards))):
            # If at final step or episode terminated, next value is 0
            if t == len(rewards) - 1 or dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            # TD error: r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE calculation: A_t = δ_t + γλA_{t+1}
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            
        # Returns = advantages + values
        returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages
    
    def _ppo_step(self, advantages, returns):
        # Convert stored experiences to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        
        for _ in range(self.ppo_epochs):
            # Get current action distribution
            means, stds = self.forward(states)
            dist = Normal(means, stds)
            
            # Get current log probabilities and entropy
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = torch.clamp(dist.entropy().mean(), min=0.05)
            
            # Ratio between new and old policies
            # Using exp(new - old) for numerical stability instead of new/old
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO-CLIP objectives
            obj1 = ratio * advantages
            obj2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(obj1, obj2).mean()
            
            # Add entropy bonus to encourage exploration
            policy_loss = policy_loss - self.entropy_coef * entropy
            
            # Value function loss
            value_preds = self.value(states).squeeze()
            value_loss = nn.MSELoss()(value_preds, returns)
            
            # Total loss
            total_loss = policy_loss + self.vf_coef * value_loss
            # Perform gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_norm)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.max_norm)
            grad_norms = list([param.grad.norm() for param in self.policy.parameters()])
            logger.info(f"Policy grad norm: [{min(grad_norms)}, {max(grad_norms)}]")
            grad_norms = list([param.grad.norm() for param in self.value.parameters()])
            logger.info(f"Value grad norm: [{min(grad_norms)}, {max(grad_norms)}]")
            
            self.optimizer.step()
            
        return policy_loss, value_loss

    def update(self):
        returns, advantages = self._process_rewards()
        
        # Perform PPO update
        policy_loss, value_loss = self._ppo_step(advantages, returns)
        total_loss = policy_loss + self.vf_coef * value_loss
        
        # Clear experience buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return total_loss.item()



class DDPGAgent(nn.Module):
    '''
    Deep Deterministic Policy Gradient
    - off-policy
    - deterministic
    '''
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'ddpg'
        self.obs_space = obs_space
        self.act_space = act_space
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.replay_batch_size = config['replay_batch_size']
        self.hidden_dim = config['hidden']
        self.up_bound = config['action_bound_up']
        self.down_bound = config['action_bound_down']
        self.max_grad_norm = config['max_grad_norm']

        #self.noise = GaussianNoise(self.act_space, self.up_bound, self.down_bound)
        self.noise = OUNoise(self.act_space, sigma=0.5)
        self.current_episode_rewards = 0
        
        self.replay_buffer = ReplayBuffer(config['buffer_size'], device=DEVICE)
        
        # Create actor and critic networks
        self.actor = build_model(config['actor_model'], obs_space, config['hidden'], act_space)
        self.critic = build_model(config['critic_model'], obs_space + act_space, config['hidden'], 1)
        
        # Create target networks
        self.target_actor = build_model(config['actor_model'], obs_space, config['hidden'], act_space)
        self.target_critic = build_model(config['critic_model'], obs_space + act_space, config['hidden'], 1)
        
        # Initialize target networks with source network parameters
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        
        # Setup optimizers - these will be set in main.py
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'], weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'], weight_decay=1e-4)

        self.to(DEVICE)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.current_episode_rewards += reward
    
    def act(self, obs, eval=False):
        """Select action based on current policy"""
        obs = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            action = self.actor(obs)
            if not eval:  # Add exploration noise during training
                noise = torch.FloatTensor(self.noise.sample(self.current_episode_rewards)).to(DEVICE)
                action += noise
            action = torch.clamp(action, min=self.down_bound, max=self.up_bound)
        return action.detach().cpu().numpy()
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.replay_batch_size:
            logger.info("REPLAY BUFFER SMALLER THAN BATCH SIZE. DECREASE BATCH SIZE.")
            return 0
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.replay_batch_size)
        self.noise.update_max_reward(rewards.mean())

        rewards = (rewards - rewards.mean())/ (rewards.std() + 1e-8)

        # Compute critic loss
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            Q_targets_next = self.target_critic(torch.cat((next_states, next_actions), dim=-1))
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        logger.info(f"Q_expected mean: {Q_expected.mean().item()} std: {Q_expected.std().item()}")
        logger.info(f"Q_targets mean: {Q_targets.mean().item()} std: {Q_targets.std().item()}")
        logger.info(f"Difference mean: {(Q_targets - Q_expected).mean().item()}, std: {(Q_targets - Q_expected).std().item()}")
        logger.info(f"Critic Loss: {critic_loss}")
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
        # grad_norms = list([param.grad.norm() for param in self.critic.parameters()])
        # logger.info(f"Critic grad norm: [{min(grad_norms)}, {max(grad_norms)}]")
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(torch.cat((states, self.actor(states)), dim=-1)).mean()
        logger.info(f"Actor Loss: {actor_loss}")
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        # grad_norms = list([param.grad.norm() for param in self.actor.parameters()])
        # logger.info(f"Critic grad norm: [{min(grad_norms)}, {max(grad_norms)}]")
        self.actor_optimizer.step()

        logger.info(f"Differential Loss:{critic_loss - actor_loss}")
        
        # Update target networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return critic_loss.item() + actor_loss.item()
    
    def _soft_update(self, target, source):
        """Soft update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def _hard_update(self, target, source):
        """Hard update of target network parameters"""
        target.load_state_dict(source.state_dict())

    def train(self, mode=True):
        self.noise.reset()
        self.current_episode_rewards = 0
        self.actor.train(mode)
        self.critic.train(mode)



class TD3Agent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        raise NotImplementedError("TD3 not yet implemented")
    
    def act(self, obs, explore=True):
        pass
    
    def update(self):
        pass
    
    def _soft_update(self, target, source):
        pass



class SACAgent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.name = 'sac'
        self.obs_dim = obs_space
        self.act_dim = act_space
        self.hidden = config['hidden']
        self.replay_batch_size = config['replay_batch_size']
        self.buffer_size = config['buffer_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.up_bound = config['action_bound_up']
        self.down_bound = config['action_bound_down']
        self.max_grad_norm = config['max_grad_norm']


        self.replay_buffer = ReplayBuffer(self.buffer_size, device=DEVICE)
        
        # Initialize networks
        self.actor = build_model(config['actor_model'], self.obs_dim, self.hidden, self.act_dim * 2).to(DEVICE)
        self.critic1 = build_model(config['critic_model'], self.obs_dim+self.act_dim, self.hidden, 1).to(DEVICE)
        self.critic2 = build_model(config['critic_model'], self.obs_dim+self.act_dim, self.hidden, 1).to(DEVICE)
        
        # Initialize target networks
        self.target_critic1 = build_model(config['critic_model'], self.obs_dim+self.act_dim, self.hidden, 1).to(DEVICE)
        self.target_critic2 = build_model(config['critic_model'], self.obs_dim+self.act_dim, self.hidden, 1).to(DEVICE)
        
        # Copy parameters to target networks
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)
        
        # Setup optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'], betas=[0.997, 0.997])
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=config['critic_lr'],
            betas = [0.997, 0.997]
        )
        
        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor([self.act_dim])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['actor_lr'])
        
        # For storing transitions
        self.transitions = ReplayBuffer(self.buffer_size, device=DEVICE)

        self.to(DEVICE)

    def forward(self, obs):
        action_params = self.actor(obs)

        action_dim = action_params.shape[-1] // 2
        means = action_params[..., :action_dim]
        log_stds = action_params[..., action_dim:]

        # "Clamp" log_stds for stability
        log_stds = torch.clamp(log_stds, -20, 2)
        stds = log_stds.exp()
        
        return means, stds

    def act(self, obs, eval=False):
        """Sample an action from normal distribution"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(DEVICE)

        with torch.no_grad():
            means, stds = self.forward(obs)
            dist = Normal(means, stds)
            action = dist.rsample()
            action = torch.clamp(action, min=self.down_bound, max=self.up_bound)
            log_prob = dist.log_prob(action).sum(-1)

        return action.detach().cpu().numpy()
    
    def get_action_and_log_prob(self, obs):
        """Get action and log probabilities - for training updates"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(DEVICE)
        
        means, stds = self.forward(obs)
        dist = Normal(means, stds)
        action = dist.rsample()
        action = torch.clamp(action, min=self.down_bound, max=self.up_bound)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        return action, log_prob
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, float(done))
    
    def update(self):
        """Update the networks using a batch from replay buffer"""
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.replay_batch_size)

        # Update critics
        with torch.no_grad():
            # Sample next actions and compute log probs
            next_actions, next_log_probs = self.actions_new, log_probs = self.get_action_and_log_prob(states)

            # Target Q-values
            q1_next = self.target_critic1(torch.cat((next_states, next_actions), dim=-1))
            q2_next = self.target_critic2(torch.cat((next_states, next_actions), dim=-1))
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Current Q-values
        q1 = self.critic1(torch.cat((states, actions), dim=-1))
        q2 = self.critic2(torch.cat((states, actions), dim=-1))
        
        # Compute critic losses
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic1_loss + critic2_loss
        
        logger.info(f"Critic Loss: {critic_loss}")

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions_new, log_probs = self.get_action_and_log_prob(states)
        q1_new = self.critic1(torch.cat((states, actions_new), dim=-1))
        q2_new = self.critic2(torch.cat((states, actions_new), dim=-1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()

        logger.info(f"Actor Loss: {actor_loss}")
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return critic_loss.item() + actor_loss.item()
    
    def _soft_update(self, target, source):
        '''Soft update of target network parameters
                (1-tau) = polyak
        '''
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def _hard_update(self, target, source):
        """Hard update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)