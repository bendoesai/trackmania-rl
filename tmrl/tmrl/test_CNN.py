from tmrl import get_environment
from time import sleep
from math import floor, sqrt
#from tmrl.custom.custom_models import conv2d_out_dims, num_flat_features, mlp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, ModuleList
import tmrl.config.config_constants as cfg


class CNN_Trackmania(nn.Module):
    def __init__(self, image_height, image_width, channels, keep_prob):
        super(CNN_Trackmania, self).__init__()

        self.IMAGE_HEIGHT = image_height
        self.IMAGE_WIDTH = image_width
        self.keep_prob = keep_prob

        # convolutional layers
        self.conv1 = nn.Conv2d(channels, 24, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=4, stride=2)

        # dropout layer
        self.dropout = nn.Dropout(keep_prob)

        # fully-connected layers
        self.fc1 = nn.Linear(64 * 1 * 1 + 9, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 3)

    def forward(self, x):
        speed, gear, rpm, images, act1, act2 = x
        images = torch.from_numpy(images).float()
        speed = torch.from_numpy(speed)
        gear = torch.from_numpy(gear)
        rpm = torch.from_numpy(rpm)
        act1 = torch.from_numpy(act1)
        act2 = torch.from_numpy(act2)

        x = F.tanh(self.conv1(images))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))

        x = self.dropout(x)

        x = x.view(-1)
        #print(x.size())
        x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)

        return x
# Instantiate the model
image_height = 64  
image_width = 64   
num_images = 4
keep_prob = 0.5     
model = CNN_Trackmania(image_height, image_width, num_images, keep_prob)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Let us retrieve the TMRL Gymnasium environment.
# The environment you get from get_environment() depends on the content of config.json
env = get_environment()

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

for i_episode in range(1, 2000):
    score = 0 #initialize episode score to 0
    obs, info = env.reset() #get state when lunar lander is restarted
    for _ in range(20000):  # rtgym ensures this runs at 20Hz by default
        #act = model(torch.from_numpy(obs[3]))  # compute action
        act = model(obs)
        act = act.detach().numpy()
        act *= 2
      
        print(act * 2)
        obs, rew, terminated, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)
        score += rew
        if terminated or truncated:
            break
    print("Episode " + str(i_episode) + ": " + str(score))
env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed