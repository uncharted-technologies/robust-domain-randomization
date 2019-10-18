"""
Trains normal, regularized, or randomized agents on the visual cartpole environment.
Networks for trained agents are saved in the results folder.
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


RANDOMIZE = True
REGULARIZE = True
LAMBDA = 10
ADD_CURRENT_STEP = True
reference_domain = (1.,1.,1.)

NUM_SEEDS = 10

color_range = np.linspace(0., 1., 5)
color_range_green = np.linspace(0.5, 1., 4)
colors =  [(r,g,b) for r in color_range for g in color_range_green for b in color_range ]
nb_steps = 100000


name = 'Normal'

if RANDOMIZE:
        name = 'Randomized'
        if REGULARIZE:
                name = 'Regularized'

for se in range(NUM_SEEDS) :
        print('seed : ', se)
        env = gym.make("CartPole-v0")
        env = CartPole_Pixel(env, RANDOMIZE, REGULARIZE, reference_domain, colors)
        env = FrameStack(env, 3)
        env.metadata['_max_episode_steps'] = 200 #useless
        env.reset()
        
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
                 log_folder = 'results/' + str(name) +'_' + str(se),
                 lam = LAMBDA,
                 regularize = REGULARIZE,
                 add_current_step=ADD_CURRENT_STEP)


        agent.learn(timesteps=nb_steps, verbose=True)
        agent.save()

        env.close()



