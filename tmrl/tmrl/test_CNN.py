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

        x = F.elu(self.conv1(images))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = self.dropout(x)

        x = x.view(-1)
        #print(x.size())
        x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x

def num_flat_features(x):
    #print("x size:")
    #print(x.size())
    size = x.size()[:-1]
    print(size)
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class VanillaCNN(Module):
    def __init__(self, q_net):
        super(VanillaCNN, self).__init__()
        self.q_net = q_net
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        hist = cfg.IMG_HIST_LEN
        print(str(self.h_out) + " " + str(self.w_out))
        self.conv1 = Conv2d(hist, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        print(str(self.h_out) + " " + str(self.w_out))
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        print(str(self.h_out) + " " + str(self.w_out))
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        print(str(self.h_out) + " " + str(self.w_out))
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        print(str(self.h_out) + " " + str(self.w_out))
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out
        self.mlp_input_features = self.flat_features + 12 if self.q_net else self.flat_features + 9
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x
        images = torch.from_numpy(images).float()
        speed = torch.from_numpy(speed)
        gear = torch.from_numpy(gear)
        rpm = torch.from_numpy(rpm)
        act1 = torch.from_numpy(act1)
        act2 = torch.from_numpy(act2)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        flat_features = num_flat_features(x)
        #print(flat_features)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
       
        print("Original sizes:")
        print("speed:", speed.size())
        print("gear:", gear.size())
        print("rpm:", rpm.size())
        print("x:", x.size())
        print("act1:", act1.size())
        print("act2:", act2.size())

        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x[0], act1, act2), -1)
        x = self.mlp(x)
        return x
    
# Instantiate the model
image_height = 64  
image_width = 64   
num_images = 4
keep_prob = 0.5     
#model = CNN_Trackmania(image_height, image_width, num_images, keep_prob)



# Let us retrieve the TMRL Gymnasium environment.
# The environment you get from get_environment() depends on the content of config.json
env = get_environment()

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

obs, info = env.reset()  # reset environment
model = VanillaCNN(False)
for _ in range(2000):  # rtgym ensures this runs at 20Hz by default
    print("Hello")
    #act = model(torch.from_numpy(obs[3]))  # compute action
    act = model(obs)
    act = act.detach().numpy()
    print(act)
    obs, rew, terminated, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)
    if terminated or truncated:
        break
env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed