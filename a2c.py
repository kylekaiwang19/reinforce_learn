import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, RMSprop
from collections import deque
from torch.distributions.categorical import Categorical
import gym

class deep_model(nn.Module):
    def __init__(self, input_units, output_units):
        super(deep_model, self).__init__()
        self.linear_1 = nn.Linear(input_units, 24)
        self.linear_2 = nn.Linear(24, 24)
        self.linear_3 = nn.Linear(24, output_units)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)

        return x


class agent:
    def __init__(self, env=''):
        self.env = gym.make('CartPole-v0')
        self.policy_net = deep_model(self.env.observation_space.shape[0], self.env.action_space.n)
        self.value_net = deep_model(self.env.observation_space.shape[0], 1)
        self.policy_optim = Adam(self.policy_net.parameters(), lr=0.001)
        self.value_optim = Adam(self.value_net.parameters(), lr=0.001)
        self.num_eps = 1000
        self.disc = 0.9
    
    def sample_action(self, obs):
        # random policy, return int
        logits = self.policy_net(obs)
        return Categorical(logits=logits).sample().item()
    
    def get_logp(self, obs, act):
        # return a tensor
        return Categorical(logits=self.policy_net(obs)).log_prob(act)
    
    def get_value(self, obs):
        # return a tensor
        return self.value_net(obs)
    
    def optim_policy(self, obs, act, rew, value):
        self.policy_optim.zero_grad()
        logp = self.get_logp(obs, act)
        loss = -(logp * (rew - value)).mean()
        loss.backward()
        self.policy_optim.step()
    
    def optim_value(self, rew, obs):
        self.value_optim.zero_grad()
        value = self.get_value(obs).view(-1)
        loss = torch.square(rew - value).mean()
        loss.backward()
        self.value_optim.step()
    
    def train_eps(self):
        eps_rew = []
        eps_obs = []
        eps_act = []
        eps_value = []
        eps_len = 0
        
        obs = self.env.reset()
        done = False
        eps_obs.append(obs)
        value = 0
        
        while not done:
            act = self.sample_action(torch.as_tensor(obs, dtype=torch.float32))
            eps_act.append(act)
            value = self.get_value(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ =  self.env.step(act)
            eps_value.append(value)
            eps_rew.append(rew + self.disc * self.get_value(torch.as_tensor(obs, dtype=torch.float32)) if not done else rew)
            if not done:
                eps_obs.append(obs)
            eps_len += 1
        
        self.optim_policy(torch.as_tensor(eps_obs, dtype=torch.float32), 
                            torch.as_tensor(eps_act, dtype=torch.float32), 
                            torch.as_tensor(eps_rew, dtype=torch.float32), 
                            torch.as_tensor(eps_value, dtype=torch.float32))
        self.optim_value(torch.as_tensor(eps_rew, dtype=torch.float32), 
                        torch.as_tensor(eps_obs, dtype=torch.float32))

        return eps_len
    
    def train(self):
        for i in range(self.num_eps):
            eps_len = self.train_eps()
            print(f'episode {i}, eposide length={eps_len}')


if __name__ == '__main__':
    a2c_agent = agent()
    a2c_agent.train()
    