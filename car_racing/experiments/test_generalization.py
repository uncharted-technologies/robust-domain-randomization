import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from carracing_wrapper import make_carracing
import gym
import numpy as np
import time
import numpy as np
import pickle
import os

from agent.ppo_simple import PPO

PATH = 'results/'
TEST_RANDOMIZE = False
RENDER = False
N_TEST_EPISODES = 20
COLOR_RANGE = [0,1]
AGENTS_TO_TEST = ['regularize']
LAMBDA_TO_TEST = 1

if TEST_RANDOMIZE:
    label = 'random'
else:
    label = 'notrandom'

class TestResults():
    def __init__(self, agent, agent_type):
        self.agent = agent
        self.agent_type = agent_type
        self.returns = []
        self.colors = []
    def add_return(self,returns,color):
        self.returns.append(returns)
        self.colors.append(color)

results = []
env = make_carracing('CarRacing-v0',COLOR_RANGE)

for dirs in os.listdir(PATH):

    info = eval(open(PATH + dirs + '/experimental-setup','r').read())

    if info['randomize']:
        agent_type = 'randomize'
    else:
        if info['lamda'] == LAMBDA_TO_TEST:
            agent_type = 'regularize'
        elif info['lamda'] == 0:
            agent_type = 'normal'
        else:
            agent_type = 'na'

    if agent_type in AGENTS_TO_TEST:

        test_results = TestResults(dirs,agent_type)    

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_std, lr, betas, gamma, K_epochs, eps_clip,randomize, lamda = info['action_std'],0.1,(0.9, 0.999),0.99,20,0,False,0

        agent = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,randomize,lamda)
        agent.policy.load_state_dict(torch.load(PATH + dirs + '/PPO_continuous.pth',map_location='cpu'))
        agent.policy_old.load_state_dict(torch.load(PATH + dirs + '/PPO_continuous.pth',map_location='cpu'))

        obs,obs_randomized = env.reset()
        returns = 0
        n_episodes = 0
        while n_episodes < N_TEST_EPISODES :
            if TEST_RANDOMIZE:
                action = agent.play(torch.FloatTensor(obs_randomized))
            else:
                action = agent.play(torch.FloatTensor(obs))
            obs, obs_randomized, rew, done, _ = env.step(action)
            if RENDER:
                env.render()
                time.sleep(0.2)
            returns += rew
            if done:
                obs,obs_randomized = env.reset()
                n_episodes += 1
                test_results.add_return(returns,env.this_episode_color)
                print('Trial {}: return of {} achieved by {} agent'.format(n_episodes, returns, agent_type))
                returns = 0

        results.append(test_results)

with open('results/generalization_results_'+label+'_l1.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)