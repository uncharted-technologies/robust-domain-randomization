"""
Loads trained agents and tests them on a plane of the RGB cube with G fixed. 
Results are saved in the results folder and plotted with plot_interpolation in the plotting folder 
"""


import pickle
import numpy as np
from env import gym
import matplotlib.pyplot as plt
import cv2

cv2.ocl.setUseOpenCL(False)

from agent.dqn import DQN
from env.visual_cartpole import CartPole_Pixel, FrameStack

from agent.cnn_cartpole import CNN_cartpole

import matplotlib
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADD_CURRENT_STEP = True


#Settings for interpolation

NUM_SEEDS = 1
# Choose colors to test on
reference_domain = (1.,1.,1.)
color_range = np.linspace(0., 1., 5)
colors = [(r,1,b) for r in color_range for b in color_range ]

# Paths to agents networks
PATH_PREFIX = 'results/networks/all_networks/big_lam100/'
agents = {'Regularized' : [PATH_PREFIX + 'Regularized_'+str(i)+'/network.pth'for i in range(NUM_SEEDS)],
'Randomized' : [PATH_PREFIX + 'Randomized_'+str(i)+'/network.pth' for i in range(NUM_SEEDS)],
'Normal' : [PATH_PREFIX + 'Normal_'+str(i)+'/network.pth'  for i in range(NUM_SEEDS)]}

env = gym.make("CartPole-v0")
env = CartPole_Pixel(env, True, False, reference_domain, colors) # Randomize = True, regularize = False for testing
env = FrameStack(env, 3)
agent = DQN(env,
         CNN_cartpole,
         replay_start_size=1000, 
         replay_buffer_size=100000, 
         gamma=0.99,
         update_target_frequency=1000,
         minibatch_size=32,
         learning_rate=1e-4,
         initial_exploration_rate=1.,
         final_exploration_rate=0.01,
         final_exploration_step=10000,
         adam_epsilon=1e-4,
         logging=True,
         loss='mse',
         lam = 0,
         regularize = False,
         add_current_step=ADD_CURRENT_STEP)


scores_all = np.zeros((3,NUM_SEEDS, len(colors))) # 3 agents

for name_id, name in enumerate(agents.keys()):
	for se in range(NUM_SEEDS):
		print('seed : ', se)
		agent.load(agents[name][se])

		for lab, col in enumerate(colors):
			env.env.colors = [col] # test on only one color
			obs, _ = env.reset()
			returns = 0
			returns_list = []
			current_step = 0.

			# average score over 1000 steps
			for i in range(1000): #10000
			    action = agent.predict(torch.FloatTensor(obs).to(device),torch.FloatTensor([current_step]).to(device))

			    obs, rew, done, info = env.step(action)
			    current_step = info['current_step']
			    returns += rew
			    if done:
			        obs, _  = env.reset()
			        returns_list.append(returns)
			        print(returns)
			        returns = 0

			print('Average score on ' + str(len(returns_list)) + ' episodes' + ' for '+ str(name) +' : ' + str(np.mean(returns_list)) + ', color : '+ str(col) )
			scores_all[name_id][se][lab] = np.mean(returns_list)

np.save('results/generalization_result_interpolation_100.npy',scores_all) # Can then be plotted in plot_extrapolation.py or plot_interpolation.py
env.close()