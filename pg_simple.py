import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.init import kaiming_normal_
from torch.optim import Adam
import gym
import numpy as np

class mlp(nn.Module):
    def __init__(self, num_input, num_output, layer_shape, act=nn.ReLU):
        super(mlp, self).__init__()
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

def train(layer_shape, num_batch=50, batch_size = 5000):
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logit_net = mlp(obs_dim, act_dim, layer_shape)
    logit_net.apply(weight_init)

    def get_policy(obs):
        logit = logit_net(obs)
        return Categorical(logits=logit)
    
    def sample_act(obs):
        return get_policy(obs).sample().item()

    def get_loss(act, obs, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    optimizer = Adam(logit_net.parameters(), lr = 1e-2)
    
    def train_one_epoch(batch_size):
        batch_acts = []
        batch_obs = []
        batch_weights = []
        batch_lens = []
        batch_ret = []
        done = False
        obs = env.reset()

        eps_ret = 0
        eps_len = 0
        while True:
            batch_obs.append(obs)
            batch_acts.append(sample_act(torch.as_tensor(obs, dtype=torch.float32)))
            obs, rew, done, _ = env.step(batch_acts[-1])
            eps_ret += rew
            eps_len += 1

            if done:
                batch_weights += [eps_ret] * eps_len
                batch_lens.append(eps_len)
                batch_ret.append(eps_ret)

                obs, done, eps_ret, eps_len = env.reset(), False, 0, 0
                
                if len(batch_acts) >= batch_size:
                    break
        
        optimizer.zero_grad()
        pg_loss = get_loss(torch.as_tensor(batch_acts, dtype=torch.float32), 
                            torch.as_tensor(batch_obs, dtype=torch.float32), 
                            torch.as_tensor(batch_weights, dtype=torch.float32))
        pg_loss.backward()
        optimizer.step()
        return pg_loss, batch_ret, batch_lens

    for i in range(num_batch):
        pg_loss, batch_ret, batch_lens = train_one_epoch(batch_size)
        print(f'epoch {i}, loss {pg_loss}, average reward {sum(batch_ret) / len(batch_ret)}, average lens {sum(batch_lens) / len(batch_lens)}')
    
    return mlp


if __name__ == '__main__':
    train([32, 32])

            
