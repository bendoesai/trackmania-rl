from tmrl import get_environment
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
from typing import Dict, Type, Set

def get_agent_required_args() -> Dict[str, Set[str]]:
    """Define required arguments for each agent type"""
    return {
        'dummy': set(),  # Dummy agent needs no config
        'vpg': {'hidden', 'batch_size', 'lr', 'gamma'},
        'trpo': {'hidden', 'batch_size', 'lr', 'gamma', 'max_kl'},
        'ppo': {'hidden', 'batch_size', 'lr', 'gamma', 'clip_ratio', 'vf_coef'},
        'ddpg': {'hidden', 'batch_size', 'lr', 'gamma', 'tau', 'buffer_size'},
        'td3': {'hidden', 'batch_size', 'lr', 'gamma', 'tau', 'buffer_size', 'policy_delay'},
        'sac': {'hidden', 'batch_size', 'lr', 'gamma', 'alpha', 'buffer_size'}
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agents on LIDAR environment')
    
    # Basic arguments
    parser.add_argument('--agent', type=str, default='dummy', 
                        choices=['dummy', 'vpg', 'trpo', 'ppo', 'ddpg', 'td3', 'sac'],
                        help='Agent type to use')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to load/save checkpoints')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluate every N episodes')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed')
    parser.add_argument('--max_episodes', type=int, default=10000,
                        help='Maximum number of episodes')
    parser.add_argument('--max_timesteps', type=int, default=100000,
                        help='Maximum timesteps per episode')
    
    # Common agent arguments
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for updates')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    
    # Agent-specific arguments
    parser.add_argument('--max_kl', type=float, default=0.01,
                        help='TRPO max KL divergence')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clip ratio')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='PPO value function coefficient')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='DDPG/TD3 soft update coefficient')
    parser.add_argument('--buffer_size', type=int, default=100000,
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

def flatten_observation(obs):
    """Flattens a tuple of tuples with varying lengths into a single tuple."""
    flat = np.concatenate([np.ravel(arr) for arr in obs])
    return flat

def evaluate(agent, env, num_episodes=5):
    agent.eval()
    eval_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = flatten_observation(obs)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action = agent.act(obs, eval=True)
            obs, reward, terminated, truncated, _ = env.step(np.array(action))
            obs = flatten_observation(obs)
            episode_reward += reward
            done = terminated or truncated
            
        eval_rewards.append(episode_reward)
    
    agent.train()
    return np.mean(eval_rewards), np.std(eval_rewards)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parse and validate config
    config = parse_args()
    validate_agent_config(config['agent'], config)
    
    # Setup environment
    env = get_environment()
    
    obs_space_flat = sum(np.prod(box.shape) for box in env.observation_space)
    num_actions = env.action_space.shape[0] * 2
    
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
    agent.optimizer = optim.Adam(
        agent.parameters(),
        lr=config['lr']
    )
    
    if config['checkpoint_path'] and Path(config['checkpoint_path']).exists():
        checkpoint = torch.load(config['checkpoint_path'])
        agent.load_state_dict(checkpoint['model'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint['episode']
        logger.info(f"Loaded checkpoint from episode {start_episode}")
    else:
        start_episode = 0
    
    # Training loop
    reward_history = []
    best_eval_reward = float('-inf')
    
    try:
        sleep(1.0)  # Allow time to focus TM20 window
        
        for episode in range(start_episode, config['max_episodes']):
            agent.train()
            obs, _ = env.reset()
            obs = flatten_observation(obs)
            episode_rewards = []
            
            # Episode loop
            for step in range(config['max_timesteps']):

                action = agent.act(obs)

                next_obs, reward, terminated, truncated, info = env.step(np.array(action))
                
                # Store transition
                agent.rewards.append(reward)
                episode_rewards.append(reward)
                
                obs = flatten_observation(next_obs)
                if terminated or truncated:
                    break
            
            total_reward = sum(episode_rewards)
            reward_history.append(total_reward)
            
            # Batch update
            if episode > 0 and episode % config['batch_size'] == 0:
                #print("Before update:", [p.norm().item() for p in agent.policy.parameters()])
                loss = agent.update()
                #print("After update:", [p.norm().item() for p in agent.policy.parameters()])

                logger.info(f"Episode {episode}, Loss: {loss:.3f}, Reward: {total_reward:.3f}")
            
            # Evaluation
            if episode % config['eval_freq'] == 0:
                mean_reward, std_reward = evaluate(agent, env)
                logger.info(f"Evaluation: Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
                
                # Save best model
                if mean_reward > best_eval_reward and config['checkpoint_path']:
                    best_eval_reward = mean_reward
                    checkpoint = {
                        'model': agent.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'episode': episode,
                        'reward': mean_reward
                    }
                    torch.save(checkpoint, config['checkpoint_path'])
                    logger.info(f"Saved new best model with reward {mean_reward:.3f}")
            
            env.unwrapped.wait()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
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
            plt.savefig('training_rewards.png')
            plt.close()

if __name__ == "__main__":
    main()