"""
Implementation of the Vanilla policy gradient algorithm
NOT GPU optimized
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random

from agent.logger import Logger
from agent.utils import set_global_seed

class VPG:

    def __init__(self,
                 env,
                 network,
                 gamma=1,
                 seed=None,
                 verbose=True,
                 learning_rate=0.01,
                 logging=True,
                 regularize=False,
                 lam=0):

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = env
        self.gamma = gamma
        self.seed = random.randint(0, 1e6) if seed is None else seed
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.logging = logging

        set_global_seed(self.seed,self.env)

        self.network = network(self.env.observation_space, self.env.action_space.n,1).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.regularize = regularize
        self.lam = lam

    @torch.no_grad()
    def act(self, state):
        network_out, _ = self.network(state)
        probs = torch.exp(F.log_softmax(network_out.squeeze(), dim=-1))
        dist = Categorical(probs)
        action = dist.sample().item()
        log_prob = torch.log(probs[action])
        return action, log_prob

    def calculate_returns(self, rewards):
        returns = torch.zeros(len(rewards))
        running_sum = 0
        for i in reversed(range(len(rewards))):
            running_sum = running_sum * self.gamma + rewards[i]
            returns[i] = running_sum

        return returns

    def learn(self, timesteps):

        if self.logging:
            logger = Logger()


        state, state_rand = self.env.reset()
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state_rand = torch.tensor(state_rand, dtype=torch.float, device=self.device)
        done = False

        states = []
        states_rand = []
        actions = []
        rewards = []
        score = 0
        episode_count = 0

        for timestep in range(timesteps):

            if not done:
                # Pick action
                action, _ = self.act(state)

                # Perform action in env
                next_state, reward, done, info = self.env.step(action)

                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)

                state_next_randomized = torch.as_tensor([0.])
                if self.regularize:
                    state_next_randomized = info['state_randomized']
                    state_next_randomized = torch.tensor(state_next_randomized, dtype=torch.float, device=self.device)


                score += reward

                # Save transitions
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                states_rand.append(state_rand)

                state = next_state
                state_rand = state_next_randomized

            if done:
                # Discount rewards and calculate advantages
                returns = self.calculate_returns(rewards)

                # Update main networks
                loss = self.train_step(states, states_rand, actions, returns)

                state, state_rand = self.env.reset()
                state = torch.tensor(state, dtype=torch.float, device=self.device)
                state_rand = torch.tensor(state_rand, dtype=torch.float, device=self.device)

                if self.verbose:
                    print("Episode {} : score {}".format(episode_count, score))
                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                    logger.add_scalar('Loss', loss, timestep)

                done = False
                states = []
                states_rand = []
                actions = []
                rewards = []
                score = 0
                episode_count += 1

        if self.logging:
            logger.save()

    def train_step(self, states, states_rand, actions, returns):
        mse = torch.nn.MSELoss()

        states = torch.stack(states).to(self.device) 
        if self.regularize:
            states_rand = torch.stack(states_rand).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        returns = returns.to(self.device)
        returns = returns.view(-1,1)


        # Get log probs
        policy_out, state_values = self.network(states)
        log_probs = F.log_softmax(policy_out, dim=-1)
        log_probs = log_probs.gather(1, actions.unsqueeze(1))

        state_values_detach = state_values.detach()  
        avantages = returns - state_values_detach


        self.optimizer.zero_grad()
        loss_policy = torch.mean(-log_probs * avantages)
        loss_value = mse(returns, state_values) 
        loss_features = 0.
        if self.regularize:
            reference_features = self.network.output_1[:-1](states.float())
            randomized_features = self.network.output_1[:-1](states_rand.float())
            loss_features = mse(reference_features, randomized_features)

        total_loss = loss_policy + loss_value + self.lam * loss_features
        
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
