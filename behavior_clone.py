import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from matplotlib import pyplot as plt

import os
import argparse
import wandb
from torch.utils.data import DataLoader, TensorDataset
from import_off_data import ImportData
from off_env import Env

def get_config():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="BCO")               # Policy name
    parser.add_argument("--env", default="outdoor off RL")        
    parser.add_argument("--batch_size", default=256, type=int)  
    parser.add_argument("--learning_rate", default=0.0001, type=float)    
    args = parser.parse_args()

    return args

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #cov2d layers NEW
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding =0) # out size 38x38x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,stride=1, padding =0) #36x36x8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #34x34x4
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,stride=2, padding =1) #16x16x4
        self.flatten= nn.Flatten()
        self.fc1 = nn.Linear(900, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.LayerNorm(int(hidden_size/2))
        self.mu = nn.Linear(int(hidden_size/2), action_size)
        self.log_std_linear = nn.Linear(int(hidden_size/2), action_size)
        
        # self.fc1 = nn.Linear(state_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)

        # self.mu = nn.Linear(hidden_size, action_size)
        # self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # print("State in act net:",state.shape)
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=self.flatten(x)
        # print('flatten shape :',x.shape)
        x=self.fc1(x)
        x=F.relu(self.bn1(x))
        x=self.fc2(x)
        x=F.relu(self.bn2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()

# init environment
config = get_config()
config.batch_size = 256
n_epoch = 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

env = Env(True,'/home/kasun/offlineRL/dataset')
train_dataset = ImportData('/media/kasun/Media/offRL/dataset/')

dataloader  = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

criterion = nn.L1Loss()
# criterion = nn.MSELoss()

config.learning_rate = 0.0001
state_size =2
action_size=3

actor_local = Actor(state_size, action_size).to(device)
actor_optimizer = optim.Adam(actor_local.parameters(), lr=config.learning_rate)



with wandb.init(project="Outdoor-BCO-offline", name="Spot outdoor", config=config):

    for itr in range(n_epoch):
        total_loss = 0
        b=0

        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience

            states = states.to(device).float()
            actions = actions.to(device).float()
            # rewards = rewards.to(device).float()
            # next_states = next_states.to(device).float()
            # dones = dones.to(device)

            actions_pred,_ = actor_local.evaluate(states) # state_trainsition_model(data)
            loss   = criterion(actions_pred, actions)
            total_loss += loss.item() 
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            b += 1

        wandb.log({
        "Epoch loss": total_loss / b,
        "Episode": itr})

        if itr % 40 == 0:
            torch.save(actor_local.state_dict(), os.path.join('./trained_models/','bco_sep14_r1_'+str(itr)+'.pkl'))

        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss / b))



