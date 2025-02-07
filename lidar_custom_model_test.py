from tmrl import get_environment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from time import sleep
from abc import ABC, abstractmethod


class RandomAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'RandomAgent'
    
    def act(self, obs):
        """
        simplistic policy for LIDAR observations
        """
        deviation = obs[1].mean(0)
        deviation /= (deviation.sum() + 0.001)
        steer = 0
        for i in range(19):
            steer += (i - 9) * deviation[i]
        steer = - np.tanh(steer * 4)
        steer = min(max(steer, -1.0), 1.0)
        return np.array([1.0, 0.0, steer])
    
class CarCrasher9000(nn.Module):
    # A2C implementation
    def __init__(self,
                 obs_space,
                 action_space,
                 hidden_size,):
        
        super(CarCrasher9000, self).__init__()

        self.name = "CarCrasher9000"
        self.action_space = action_space
        self.obs_space = obs_space

        torch.set_default_tensor_type(torch.FloatTensor)

        self.critic = nn.Sequential(
            nn.Linear(self.obs_space, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size/2), 1)
        )

        self.actor_base = nn.Sequential(
            nn.Linear(self.obs_space, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.LeakyReLU(),
        )

        self.actor_mu = nn.Sequential(
            nn.Linear(int(hidden_size/2), self.action_space),
            nn.Tanh() #compress outputs between 1 and -1
        )

        self.actor_var = nn.Sequential(
            nn.Linear(int(hidden_size/2), self.action_space),
            nn.Softplus()
        )

    def forward(self, obs):
        
        obs = obs.float()

        value = self.critic(obs)

        action_base = self.actor_base(obs)
        action_mu = self.actor_mu(action_base)
        action_var = self.actor_var(action_base)


        return value, torch.stack((action_mu, action_var), dim=1)

    def act(self, obs):
        #value of the action, mean and standard deviation of
        #each action slot (means, variances)
        value, dists = self.forward(obs)

        #output should be (3,)
        action = [np.random.normal(float(dist[0]), float(dist[1])) for dist in dists]
        for i, a in enumerate(action):
            if a > 1:
                action[i]=0.999
            elif a < -1:
                action[i]=-0.999
        return action, value

def obs_to_tensor(obs):
    return torch.tensor(np.hstack([obs[0], np.hstack(obs[1]), np.squeeze(obs[2]), np.squeeze(obs[3])]), dtype=torch.float32)

if __name__ == "__main__":
    # Let us retrieve the TMRL Gym environment.
    # The environment you get from get_environment() depends on the content of config.json
    
    try:
        MAX_TIMESTEPS = 50000
        MAX_EPISODES = 10000

        hyperparameters = {
            'lr' : 5e-4,
            'gamma': 0.99,
        }
        
        env = get_environment()

        agent = CarCrasher9000(
            obs_space = 83,
            action_space = 3,
            hidden_size = 512,
        )

        opt = torch.optim.Adam(
            agent.parameters()
        )

        #agent = RandomAgent()

        reward_tracker = []
        time_tracker = []
        high_score = 0

        sleep(1.0)  # just so we have time to focus the TM20 window after starting the script
        for episode in range(MAX_EPISODES):
            rewards = []
            values = []
            # default LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,))
            # representing: (speed, 4 last LIDARs, 2 previous actions)
            # actions are [gas, break, steer], analog between -1.0 and +1.0
            obs, _ = env.reset()  # reset environment
            for step in range(MAX_TIMESTEPS):  # rtgym ensures this runs at 20Hz by default
                # compute action
                action, value = agent.act(obs_to_tensor(obs))
                values.append(value.detach().numpy())

                # apply action (rtgym ensures healthy time-steps)
                next_obs, reward, terminated, truncated, info = env.step(np.array(action))
                
                rewards.append(reward)

                obs = next_obs
                
                if terminated or truncated:
                    action, q_val = agent.act(obs_to_tensor(obs))
                    q_val = q_val.detach().numpy()
                    reward_tracker.append(np.sum(rewards))
                    if np.sum(rewards) > high_score:
                        torch.save(agent.state_dict(), 'checkpoints/LIDAR_{}'.format(np.sum(rewards)))
                        high_score=np.sum(rewards)+1
                    time_tracker.append(step)
                    print("episode: {}, reward: {}, total length: {} \n".format(episode, np.sum(rewards), step))
                    break

            opt.step()
            env.unwrapped.wait()  # rtgym-specific method to artificially 'pause' the environment when needed
    
    except KeyboardInterrupt:
        smoothed_rewards = pd.Series.rolling(pd.Series(reward_tracker), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        plt.plot(reward_tracker)
        plt.plot(smoothed_rewards)
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

        plt.plot(time_tracker)
        plt.xlabel('Episode')
        plt.ylabel('Episode length')
        plt.show()