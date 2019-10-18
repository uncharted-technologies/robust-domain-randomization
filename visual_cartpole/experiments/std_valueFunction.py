"""
Loads trained agents from the results folder and measures variance of estimated Q values
on different background colors, which we use for figure 5.
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


ADD_CURRENT_STEP = True
reference_domain = (1.,1.,1.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SEEDS = 10
PATH_PREFIX = 'results/networks/all_networks/big_lam10/'
DIVIDE_STD_BY_MEAN = False # divide std by mean value ?

# Choose colors to calculate std on

color_range = np.linspace(0., 1., 5)
color_range_green = np.linspace(0.5, 1., 4)
colors = [(r,g,b) for r in color_range for g in color_range_green for b in color_range ]

# Paths to agents networks
agents = {'Normal' : [PATH_PREFIX + 'Normal_'+str(i)+'/network.pth'  for i in range(NUM_SEEDS)],
'Regularized' : [PATH_PREFIX + 'Regularized_'+str(i)+'/network.pth'for i in range(NUM_SEEDS)],
'Randomized' : [PATH_PREFIX + 'Randomized_'+str(i)+'/network.pth' for i in range(NUM_SEEDS)]}


env = gym.make("CartPole-v0")
env = CartPole_Pixel(env, False, False, reference_domain, colors) # Randomize = False, regularize = False for std calculation
env = FrameStack(env, 3, get_all_randomized_per_step = True)
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



agents_std = np.zeros((3,NUM_SEEDS))

for name_id, name in enumerate(agents.keys()):
	for se in range(NUM_SEEDS):
		print('seed : ', se)
		values = []

		agent.load(agents[name][se])

		obs, rand_obs = env.reset()
		returns = 0
		returns_list = []
		current_step = 0.
		for i in range(300): 
			for lab,rand_ob in enumerate(rand_obs):
				values.append(agent.network(torch.FloatTensor(rand_ob).to(device), torch.FloatTensor([current_step]).to(device)).max().item())

			action = agent.predict(torch.FloatTensor(obs).to(device), torch.FloatTensor([current_step]).to(device))

			obs, rew, done, info = env.step(action)
			rand_obs = info['state_randomized']
			current_step = info['current_step']

			returns += rew
			if done:
				obs, rand_obs = env.reset()
				returns_list.append(returns)
				returns=0
				break
				
		std = np.mean([np.std(values[k*len(colors):(k+1)*len(colors)]) for k in range(i)])
		if DIVIDE_STD_BY_MEAN :
			std = np.mean([np.std(values[k*len(colors):(k+1)*len(colors)])
				/np.mean(values[k*len(colors):(k+1)*len(colors)])
				 for k in range(i)])
		print('std for ' + str(name) + ' : '+  str(std))

		agents_std[name_id][se] = std

np.save('results/std_valueFunction.npy',agents_std)
env.close()