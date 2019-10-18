"""
This code is inspired by https://github.com/nikhilbarhate99/PPO-PyTorch
"""


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_randomized = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_randomized[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(6, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.output = nn.Sequential(nn.Linear(4096, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, action_dim+1))
                        

        self.conv.apply(lambda x: init_weights(x, 1))
        self.output.apply(lambda x: init_weights(x, 1))

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def get_features(self,obs,obs_randomized):
        """
        Get activations of last hidden layer for both normal and regularized observations
        """

        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = obs/255
        obs = self.conv(obs)
        obs = obs.view(obs.size(0), -1)
        features = self.output[:-1](obs)

        if len(obs_randomized.shape) != 4:
            obs_randomized = obs_randomized.unsqueeze(0)
        obs_randomized = obs_randomized.permute(0,3,1,2)
        obs_randomized = obs_randomized/255
        obs_randomized = self.conv(obs_randomized)
        obs_randomized = obs_randomized.view(obs_randomized.size(0), -1)
        features_randomized = self.output[:-1](obs_randomized)

        return features, features_randomized
        
    def forward(self,obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = obs/255
        obs = self.conv(obs)
        obs = obs.view(obs.size(0), -1)
        raw_output = self.output(obs)
        return raw_output[:,:-1].tanh(),raw_output[:,-1]
    
    def act(self, state, state_randomized, memory, randomize):
        if randomize:
            action_mean = self.forward(state_randomized)[0]
        else:
            action_mean = self.forward(state)[0]
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.states_randomized.append(state_randomized)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, state_randomized, action, randomize):   

        if randomize:
            actor_output,critic_output = self.forward(state_randomized)
        else:
            actor_output,critic_output = self.forward(state)

        action_mean = torch.squeeze(actor_output)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = critic_output
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,randomize, lamda):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.randomize = randomize
        self.lamda = lamda
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, state_randomized, memory):
        state = torch.FloatTensor(state).to(device)
        state_randomized = torch.FloatTensor(state_randomized).to(device)
        return self.policy_old.act(state, state_randomized, memory,self.randomize).cpu().data.numpy().flatten()

    def play(self, state):
        action_mean = self.policy.forward(state)[0]
        cov_mat = torch.diag(self.policy.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        
        return action.data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_states_randomized = torch.squeeze(torch.stack(memory.states_randomized).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_states_randomized, old_actions, self.randomize)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            if self.lamda != 0:
                features, features_randomized = self.policy.get_features(old_states, old_states_randomized)
                loss += self.lamda * F.mse_loss(features, features_randomized, reduction='mean')

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
