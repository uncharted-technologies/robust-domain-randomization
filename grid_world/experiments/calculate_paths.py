"""
Use this code to train both regularized and randomized agents and see how often they choose the same path

"""


import time
import random
import pickle

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from agent.mlp import MLP_Multihead
from agent.vpg import VPG
from gridworld import GridworldEnv

# Randomized agent : RANDOMIZE = True, Regularized = False, INITIAL_ADDITIONAL_PARAMS = [0, 0] to add two 'noise' inputs for example
# Regularized agent : RANDOMIZE = True, Regularized = True, INITIAL_ADDITIONAL_PARAMS = [0, 0] to add two 'noise' inputs for example
# Normal agent : RANDOMIZE = False, Regularized = False, INITIAL_ADDITIONAL_PARAMS = []
RANDOMIZE = True  
REGULARIZE = True
LAMBDA = 1
INITIAL_ADDITIONAL_PARAMS = [5]
RANDOMIZATION_SPACE = [ -5, 5]
GOAL_REWARD = 1
LAVA_REWARD = -1
STEP_REWARD = 0
OUT_OF_GRID_REWARD = -1

N_SEEDS = 100
same_actions = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = ''
if RANDOMIZE :
  NAME = 'rand'
  if REGULARIZE:
  	NAME = 'reg'

for se in range(N_SEEDS):
  print('seed : ' + str(se))
  first_actions = []
  env = GridworldEnv(randomized_params = INITIAL_ADDITIONAL_PARAMS,
 randomize = RANDOMIZE,
  regularize = REGULARIZE, 
  randomization_space = RANDOMIZATION_SPACE,
  goal_reward = GOAL_REWARD,
  lava_reward = LAVA_REWARD,
  step_reward = STEP_REWARD,
  out_of_grid = OUT_OF_GRID_REWARD,
  max_episode_steps = 10)

  nb_steps = 4000

  agent = VPG(env,
            MLP_Multihead,
            gamma=1,
            verbose=False,
            learning_rate=1e-3,
            regularize = REGULARIZE,
            lam = LAMBDA)
  print(agent.seed)


  agent.learn(timesteps=nb_steps)

  obs, _ = env.reset()
  scores = []
  for rp in RANDOMIZATION_SPACE :
    first_action = 0
    env = GridworldEnv(randomized_params = [rp],
 randomize = True,
  regularize = False, 
  randomization_space = [rp],
  goal_reward = GOAL_REWARD,
  lava_reward = LAVA_REWARD,
  step_reward = STEP_REWARD,
  out_of_grid = OUT_OF_GRID_REWARD,
  max_episode_steps = 10)

    obs,_ = env.reset()
    score = 0

    for i in range(10000):
      action, _ = agent.act(torch.FloatTensor(obs).to(device))
      if i == 0:
        first_action = action
      obs, rew, done, info = env.step(action)
      score += rew
      if done:
        obs,_ = env.reset()
        break

    first_actions.append(action)
    scores.append(score)
  print(scores, first_actions)

  if all(np.array(scores)>=0):
    same_actions.append(int(all(elem == first_actions[0] for elem in first_actions)))
  
np.save('results/' + str(NAME)+'_vpg.npy',np.array(same_actions))
plt.hist(same_actions)
plt.title(NAME + ' : ' + str(sum(same_actions)))
plt.show()
  
