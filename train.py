from tmrl import get_environment
import gymnasium as gym

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import agents
from time import sleep
import argparse
from pathlib import Path
import logging
import traceback
import sys
from typing import Dict, Type, Set

from utils import flatten_and_norm

def get_agent_required_args() -> Dict[str, Set[str]]:
    """Define required arguments for each agent type"""
    #TODO: implement at agent level
    return {
        'dummy': set(),  # Dummy agent needs no config
        'vpg': {'hidden', 'batch_size', 'lr', 'gamma'},
        'trpo': {'hidden', 'batch_size', 'lr', 'gamma', 'max_kl'},
        'ppo': {'hidden', 'batch_size', 'actor_lr', 'critic_lr', 'gamma', 'clip_ratio', 'vf_coef'},
        'ddpg': {'hidden', 'batch_size', 'actor_lr', 'critic_lr', 'gamma', 'tau', 'buffer_size'},
        'td3': {'hidden', 'batch_size', 'lr', 'gamma', 'tau', 'buffer_size', 'policy_delay'},
        'sac': {'hidden', 'batch_size', 'actor_lr', 'critic_lr', 'gamma', 'alpha', 'buffer_size'}
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents on LIDAR environment')
    
    # Basic arguments
    parser.add_argument('--agent', type=str, default='dummy', 
                        choices=['dummy', 'vpg', 'trpo', 'ppo', 'ddpg', 'td3', 'sac'],
                        help='Agent type to use')
    parser.add_argument('--actor_model', type=str, default='basicnet', 
                        choices=['basicnet'],
                        help='actor model type to use')
    parser.add_argument('--critic_model', type=str, default='basicnet', 
                        choices=['basicnet'],
                        help='critic model type to use')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to load/save checkpoints')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluate every N episodes')
    parser.add_argument('--report_freq', type=int, default=10,
                        help='Report every N episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_episodes', type=int, default=10000,
                        help='Maximum number of episodes')
    parser.add_argument('--max_timesteps', type=int, default=100000,
                        help='Maximum timesteps per episode')
    

    parser.add_argument('--easy_env', action='store_true',
                        help='test algorithm with Mountain Car Continuous')
    parser.add_argument('--test_only', action='store_true',
                        help='Only test the agent (no training)')
    parser.add_argument('--test_episodes', type=int, default=5,
                        help='Number of episodes to test when in test mode')


    # Common agent arguments
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for updates')
    parser.add_argument('--actor_lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--critic_lr', type=float, default=1,
                        help='critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--action_bound_down', type=float, default=-np.inf,
                        help='lower action bound')
    parser.add_argument('--action_bound_up', type=float, default=np.inf,
                        help='upper action bound')
    parser.add_argument('--max_grad_norm', type=float, default=1,
                        help='max gradient value')
    
    # Agent-specific arguments
    parser.add_argument('--max_kl', type=float, default=0.01,
                        help='TRPO max KL divergence')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clip ratio')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='PPO value function coefficient')
    parser.add_argument('--entropy', type=float, default=0.01,
                        help='PPO entropy coefficient')
    parser.add_argument('--lambda', type=float, default=0.95,
                        help='PPO GAE coefficient')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='PPO train epochs')
    parser.add_argument('--replay_batch_size', type=int, default=128,
                        help='Replay Buffer sample size')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='DDPG/TD3 soft update coefficient')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='DDPG/TD3/SAC replay buffer size')
    parser.add_argument('--policy_delay', type=int, default=2,
                        help='TD3 policy update delay')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='SAC entropy coefficient')
    
    args = parser.parse_args()
    
    # Convert args to dict for easier handling
    config = vars(args)
    return config

def validate_agent_config(agent_type: str, config: dict) -> None:
    """Validate config based on agent type"""
    #TODO: implement at agent level (_check_params)
    if config['test_only'] and config['checkpoint_path']:
        return

    required_args = get_agent_required_args()[agent_type]
    missing = [arg for arg in required_args if arg not in config or config[arg] is None]
    if missing:
        raise ValueError(f"Agent {agent_type} requires the following arguments: {missing}")
    
    # Agent-specific validation
    if agent_type == 'vpg':
        if config['batch_size'] < 1:
            raise ValueError("batch_size must be positive")
        if config['lr'] <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= config['gamma'] <= 1:
            raise ValueError("gamma must be between 0 and 1")
    
    elif agent_type == 'ppo':
        if config['clip_ratio'] <= 0:
            raise ValueError("clip_ratio must be positive")
        if config['vf_coef'] < 0:
            raise ValueError("vf_coef must be non-negative")

def build_agent(agent_name: str, config: dict, obs_space_flat: int, num_actions: tuple):
    agent_map = {
        'dummy': agents.DummyAgent,
        'vpg': agents.VPGAgent,
        'trpo': agents.TRPOAgent,
        'ppo': agents.PPOAgent,
        'ddpg': agents.DDPGAgent,
        'td3': agents.TD3Agent,
        'sac': agents.SACAgent
    }

    if agent_name.lower() not in agent_map:
        raise ValueError(f'Agent {agent_name} not recognized. Choose from {list(agent_map.keys())}')
    
    return agent_map[agent_name.lower()](config, obs_space_flat, num_actions)

def evaluate(agent, env, num_episodes=5):
    agent.eval()
    eval_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = flatten_and_norm(obs)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = agent.act(obs, eval=True)
            obs, reward, terminated, truncated, _ = env.step(np.array(action))
            obs = flatten_and_norm(obs)
            episode_reward += reward
            done = terminated or truncated
            
        eval_rewards.append(episode_reward)
    
    agent.train()
    return np.mean(eval_rewards), np.std(eval_rewards)

def main():
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Parse and validate config
    config = parse_args()
    validate_agent_config(config['agent'], config)
    
    # Setup environment
    if config['easy_env']:
        if config['test_only']:
            env = gym.make("MountainCarContinuous-v0", render_mode='human')
        else:
            env = gym.make("MountainCarContinuous-v0")
    else:
        env = get_environment()
    
    print(env.observation_space)
    if config['easy_env']:
        obs_space_flat = env.observation_space.shape[0]
    else:
        obs_space_flat = sum(np.prod(box.shape) for box in env.observation_space)
    
    num_actions = env.action_space.shape[0]

    logger.info(f"Observation space: {obs_space_flat}")
    logger.info(f"Action space: {num_actions}")
    logger.info(f"Training {config['agent']} agent with config:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Initialize agent and optimizer
    agent = build_agent(config['agent'], config, obs_space_flat, num_actions)
    
    file_name = os.path.join(config['checkpoint_path'], agent.name)
    if config['easy_env']:
        file_name = str(file_name) + "_easy"

    checkpoint = None

    if config['checkpoint_path'] and os.path.exists(file_name + "_checkpoint.pth"):
        checkpoint = torch.load(file_name + "_checkpoint.pth", weights_only=False)
        agent.load_state_dict(checkpoint['model'])
        
        # Load optimizer states based on agent type
        if hasattr(agent, 'actor_optimizer') and hasattr(agent, 'critic_optimizer'):
            if 'actor_optimizer' in checkpoint and 'critic_optimizer' in checkpoint:
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            else:
                logger.warning("Checkpoint doesn't contain separate optimizer states for actor and critic")
        elif hasattr(agent, 'optimizer') and 'optimizer' in checkpoint:
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        
        start_episode = checkpoint.get('episode', 0)
        logger.info(f"Loaded checkpoint from episode {start_episode}")
    else:
        start_episode = 0

    if config['test_only']:
        logger.info("Running in test-only mode")
        mean_reward, std_reward = evaluate(agent, env, config['test_episodes'])
        logger.info(f"Evaluation: Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
        env.close()
        return 0
    
    # Training loop
    reward_history = []
    best_eval_reward = checkpoint['reward'] if checkpoint else float('-inf')
    timesteps = 0
    
    try:
        sleep(1.0)  # Allow time to focus TM20 window
        
        for episode in range(start_episode, config['max_episodes']):
            agent.train()
            obs, _ = env.reset()

            obs = flatten_and_norm(obs)
            episode_rewards = []
            
            # Episode loop
            for step in range(config['max_timesteps']):
                action = agent.act(obs)

                next_obs, reward, terminated, truncated, info = env.step(np.array(action))
                
                agent.store_transition(obs, action, reward, flatten_and_norm(next_obs), terminated or truncated)
                episode_rewards.append(reward)
                
                obs = flatten_and_norm(next_obs)
                if terminated or truncated:
                    timesteps += step
                    break
            
            #print(np.mean(episode_rewards))
            total_reward = sum(episode_rewards)
            reward_history.append(total_reward)
            
            # Batch update
            if episode > start_episode and episode % config['batch_size'] == 0:
                #print("Before update:", [p.norm().item() for p in agent.policy.parameters()])
                loss = agent.update()
                #print("After update:", [p.norm().item() for p in agent.policy.parameters()])

                logger.info(f"Episode {episode}, Timesteps {timesteps}, Loss: {loss:.3f}, Reward: {total_reward:.3f}")

            # Evaluation
            if episode > start_episode and episode % config['eval_freq'] == 0:
                mean_reward, std_reward = evaluate(agent, env, config['test_episodes'])
                logger.info(f"Evaluation: Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

                # Save best model
                if mean_reward > best_eval_reward and config['checkpoint_path']:
                    best_eval_reward = mean_reward
    
                    # Create base checkpoint with model state and metadata
                    checkpoint = {
                        'model': agent.state_dict(),
                        'episode': episode,
                        'reward': mean_reward
                    }
                    
                    # Handle different optimizer configurations
                    if hasattr(agent, 'actor_optimizer') and hasattr(agent, 'critic_optimizer'):
                        # For agents with separate optimizers like DDPG
                        checkpoint['actor_optimizer'] = agent.actor_optimizer.state_dict()
                        checkpoint['critic_optimizer'] = agent.critic_optimizer.state_dict()
                    elif hasattr(agent, 'optimizer'):
                        # For agents with a single optimizer
                        checkpoint['optimizer'] = agent.optimizer.state_dict()
                    
                    # Generate appropriate filename if not specified
                    if not config['checkpoint_path'].endswith('.pth'):
                        file_path = os.path.join(f"{file_name}_checkpoint.pth")
                    else:
                        file_path = config['checkpoint_path']
                        
                    torch.save(checkpoint, file_path)
                    logger.info(f"Saved new best model with reward {mean_reward:.3f}")
            
            if not config['easy_env']:
                env.unwrapped.wait()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        print("Error during training:", e)
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info

    finally:
        # Cleanup and plotting
        env.close()
        
        if reward_history:
            # Plot training curves
            smoothed_rewards = pd.Series(reward_history).rolling(10).mean()
            plt.figure(figsize=(10, 5))
            plt.plot(reward_history, alpha=0.6, label='Raw')
            plt.plot(smoothed_rewards, label='Smoothed')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()

            plt.savefig(file_name + '_training_rewards.png')
            plt.close()

if __name__ == "__main__":
    main()