"""
Train agents on xi=5 and test them on xi=-5, for different values of lambda. 
Results are saved in the results folder and can be plotted with the script in the plotting folder.
"""


import time
import random
import pickle

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


from agent.mlp import MLP_Multihead
from agent.vpg import VPG
from gridworld import GridworldEnv

# Randomized agent : RANDOMIZE = True, Regularized = False, INITIAL_ADDITIONAL_PARAMS = [0, 0] to add two 'noise' inputs for example
# Regularized agent : RANDOMIZE = True, Regularized = True, INITIAL_ADDITIONAL_PARAMS = [0, 0] to add two 'noise' inputs for example
# Normal agent : RANDOMIZE = False, Regularized = False, INITIAL_ADDITIONAL_PARAMS = []
RANDOMIZE = True  
REGULARIZE = True
INITIAL_ADDITIONAL_PARAMS = [5]
RANDOMIZATION_SPACE = [ 5, -5]
GOAL_REWARD = 1
LAVA_REWARD = -1
STEP_REWARD = 0
OUT_OF_GRID_REWARD = -1

TEST_DOMAIN = -5


N_TRAIN_SEEDS = 100
N_ROLLOUTS = 100
LAMDA_RANGE = np.geomspace(0.00000001,0.01,30)

NAME = ''
if RANDOMIZE :
  NAME = 'rand'
  if REGULARIZE:
  	NAME = 'reg'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_scores = np.zeros((len(LAMDA_RANGE),N_TRAIN_SEEDS))
test_scores = np.zeros((len(LAMDA_RANGE),N_TRAIN_SEEDS,N_ROLLOUTS))
lipschitz_constants = np.zeros((len(LAMDA_RANGE),N_TRAIN_SEEDS))

for idx,lamda in enumerate(LAMDA_RANGE):

  print(lamda)
  se = 0

  while se < N_TRAIN_SEEDS:

    print('lambda: ' + str(lamda),'seed : ' + str(se))
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
              lam = lamda)

    agent.learn(timesteps=nb_steps)


    #Test on train environment

    scores = []
    for test_rollout in range(N_ROLLOUTS):
      

      env = GridworldEnv(randomized_params = [RANDOMIZATION_SPACE[0]],
            randomize = True,
            regularize = False, 
            randomization_space = [RANDOMIZATION_SPACE[0]],
            goal_reward = GOAL_REWARD,
            lava_reward = LAVA_REWARD,
            step_reward = STEP_REWARD,
            out_of_grid = OUT_OF_GRID_REWARD,
            max_episode_steps = 10)

      obs,_ = env.reset()
      score = 0

      done = False
      while not done:
        action, _ = agent.act(torch.FloatTensor(obs).to(device))
        obs, rew, done, info = env.step(action)
        score += rew
        if done:
          obs,_ = env.reset()
      
      scores.append(score)



    if np.mean(scores) >= 0.5:

      train_scores[idx,se] = np.mean(scores)

      #Calculate Lipschitz constant
      with torch.no_grad():
        grid_train = [[x,y,RANDOMIZATION_SPACE[0]] for x in range(3) for y in range(3)][:-1]

        all_states_train = torch.FloatTensor(grid_train)
        network_train = agent.network(all_states_train)[0]
        probs_train = torch.exp(F.log_softmax(network_train.squeeze(), dim=-1))

        grid_test = [[x,y,RANDOMIZATION_SPACE[1]] for x in range(3) for y in range(3)][:-1]

        all_states_test = torch.FloatTensor(grid_test)
        network_test = agent.network(all_states_test)[0]
        probs_test = torch.exp(F.log_softmax(network_test.squeeze(), dim=-1))

      total_variation_distance = torch.max(0.5 * torch.sum(torch.abs(probs_train-probs_test),dim=1))
      lipschitz_constants[idx,se] = total_variation_distance/abs(RANDOMIZATION_SPACE[0] - RANDOMIZATION_SPACE[1])

      #Calculate score
      for rollout in range(N_ROLLOUTS):

        env = GridworldEnv(randomized_params = [RANDOMIZATION_SPACE[1]],
            randomize = True,
            regularize = False, 
            randomization_space = [RANDOMIZATION_SPACE[1]],
            goal_reward = GOAL_REWARD,
            lava_reward = LAVA_REWARD,
            step_reward = STEP_REWARD,
            out_of_grid = OUT_OF_GRID_REWARD,
            max_episode_steps = 10)

        obs,_ = env.reset()
        score = 0

        done = False
        while not done:
          action, _ = agent.act(torch.FloatTensor(obs).to(device))
          obs, rew, done, info = env.step(action)
          score += rew
          if done:
            obs,_ = env.reset()

        test_scores[idx,se,rollout] = score



      se += 1

    else:
      print("training failed, retrying ...")
  
np.save('results/'+str(NAME)+'_scores_train_vpg.npy',train_scores)
np.save('results/'+str(NAME)+'_scores_vpg.npy',test_scores)
np.save('results/'+str(NAME)+'_lipschitz_vpg.npy',lipschitz_constants)
print(test_scores)
  
