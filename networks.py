import torch.nn as nn

class basicnet(nn.Module):
    def __init__(self, obs_space, hidden, act_space, hidden_depth = 1):
        super().__init__()
        
        self.name = 'basicnet'
        self.network = nn.Sequential(
            nn.Linear(obs_space, hidden),
            nn.LeakyReLU())
        
        for i in range(hidden_depth):
            self.network.extend(nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(),
            ))
        
        self.network.append(
            nn.Linear(hidden, act_space)
        )


    def forward(self, obs):
        action_params = self.network(obs)
        return action_params
    

class LSTMNet(nn.Module):
    pass