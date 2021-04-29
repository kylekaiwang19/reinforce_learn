import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.optim import Adam, RMSprop
from collections import deque
import gym

class dqn(nn.Module):
    def __init__(self, num_input, num_output, layer_shape, act=nn.ReLU):
        super(dqn, self).__init__()
        layer_shape = [num_input] + layer_shape
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_shape)):
            self.layers.append(nn.Linear(layer_shape[i-1], layer_shape[i]))
            self.layers.append(act())
        self.layers.append(nn.Linear(layer_shape[-1], num_output))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def weight_init(layer):
    if isinstance(layer, nn.Linear):
        kaiming_normal_(layer.weight)

MAX_MEMORY = 20000
TRAIN_START = 1000
class DQN_agent:
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEMORY)
    
    def add_sample(self, sample):
        self.memory.append(sample)
    
    def get_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)


def train(layer_shape, activation = nn.ReLU, num_epochs=500, batch_size=20000,sample_size=64, dis_fact=0.99):
    env = gym.make('CartPole-v0')
    num_obs = env.observation_space.shape[0]
    num_acts = env.action_space.n

    agent = DQN_agent()

    policy_net = dqn(num_obs, num_acts, layer_shape)
    target_net = dqn(num_obs, num_acts, layer_shape)
    policy_net.apply(weight_init)
    target_net.load_state_dict(policy_net.state_dict())

    def pred_policy(obs):
        return policy_net(torch.as_tensor(obs, dtype=torch.float32))

    def pred_target(obs):
        return target_net(torch.as_tensor(obs, dtype=torch.float32))

    def sample_act(i, obs):
        eps = make_eps(i, num_epochs)
        if random.random() < eps:
            act = random.randint(0, num_acts-1)
        else:
            with torch.no_grad():
                act = torch.argmax(pred_policy(obs)).item()
        return act

    def make_eps(i, total_epochs):
        return max(0.99 - i * (0.99-0.01) / 5000, 0.01)

    def calc_loss(obs, act, rew, new_obs, done):
        target = torch.where(done==True, rew, rew + dis_fact * torch.max(pred_target(new_obs), dim=1)[0]).detach()
        q_pred = pred_policy(obs).gather(1, act.view(-1, 1))
        return F.mse_loss(q_pred.view(q_pred.shape[0]), target)

    optim = Adam(policy_net.parameters(), lr=1e-3)

    def train_one_epoch(step):
        # for comparing performance
        total_rew = []
        total_len = []
        done = False
        obs = env.reset()
        eps_ret = 0
        eps_len = 0

        def optimize_model():
            if agent.size() >= TRAIN_START:
                sample = agent.get_batch(sample_size)
                obs, act, rew, new_obs, done = zip(*sample)
                optim.zero_grad()
                loss = calc_loss(torch.as_tensor(obs, dtype=torch.float32),
                        torch.as_tensor(act, dtype=torch.int64),
                        torch.as_tensor(rew, dtype=torch.float32),
                        torch.as_tensor(new_obs, dtype=torch.float32),
                        torch.as_tensor(done, dtype=torch.bool))
                loss.backward()
                optim.step()

        while not done:
            act = sample_act(step, obs)
            step += 1
            new_obs, rew, done, _ = env.step(act)
            agent.add_sample((obs, act, rew, new_obs, done))

            obs = new_obs
            eps_ret += rew
            eps_len += 1
            # every iteration, train one time
            optimize_model()
        total_rew.append(eps_ret)
        total_len.append(eps_len)
        target_net.load_state_dict(policy_net.state_dict())
        return total_rew, total_len, step
    
    step = 0
    for i in range(num_epochs):
        epoch_rew, epoch_len, step = train_one_epoch(step)
        print(f'epoch number {i+1}, aveage reward is {sum(epoch_rew) / len(epoch_rew)}, memory size {agent.size()}')

if __name__ == '__main__':
    train([24, 24], num_epochs=500, batch_size=1000)




        





