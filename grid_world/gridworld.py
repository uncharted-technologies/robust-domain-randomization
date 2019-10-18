import time
import random
import pickle

import gym
import sys
import os
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt

 
# CONVENTION : 0 = BACKGROUND, 1 = LAVA, 4 = AGENT, 3 = GOAL, 2 = some object to collet, not used currently

# define colors
# 0: white BACKGROUND; 1 : red LAVA ; 2 : gray; 3 : green GOAL; 4 : blue AGENT
COLORS = {0:[1.0,1.0,1.0], 1:[1., 0.,0.], \
          2:[0.5,0.5,0.5], 3:[0.0,1.0,0.0], \
          4:[0.,0.2,1.], 5:[1.0,0.0,1.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0]} # Ignore 6 and 7, useful only for rendering

class GridworldEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	num_env = 0 
	def __init__(self, randomized_params = [], 
		randomize = False, 
		regularize = False, 
		randomization_space = [0], 
		goal_reward = 10, 
		lava_reward = -2, 
		step_reward = -1,
		out_of_grid = -10, 
		seed = 1337,
		max_episode_steps = 10):

		#self._seed = 0
		self.actions = [0, 1]#[0, 1, 2, 3, 4]
		#self.inv_actions = [0, 2, 1, 4, 3]
		self.action_space = spaces.Discrete(2)#spaces.Discrete(5)
		self.action_pos_dict = {0: [1,0], 1: [0,1]}#{0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
		self.max_episode_steps = max_episode_steps
		self.current_step = 0

		''' Reward parameters '''
		self.goal_reward = goal_reward
		self.lava_reward = lava_reward
		self.step_reward = step_reward
		self.out_of_grid = out_of_grid

		''' RNG '''
		self.seed(seed=seed)
		
		''' Randomization parameters '''
		self.randomization_space = randomization_space
		self.randomized_params = randomized_params
		self.regularized_params = copy.deepcopy(randomized_params)
		self.randomize = randomize
		self.regularize = regularize


		''' initialize system state ''' 
		this_file_path = os.path.dirname(os.path.realpath(__file__))
		self.grid_map_path = os.path.join(this_file_path, 'plan.txt')        
		self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
		self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
		self.grid_map_shape = self.start_grid_map.shape
		self.obs_shape = [self.grid_map_shape[0], self.grid_map_shape[1], 3]  # shape of the image for the rendering
		self.observation = self._gridmap_to_observation(self.start_grid_map)
		

		''' set observation space '''
		self.observation_space = spaces.Box(low=0, high=max(self.grid_map_shape[0], self.grid_map_shape[1]), shape=(2 + len(self.randomized_params),), dtype=np.int)

		''' agent state: start, target, current state '''
		self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state(self.start_grid_map)
		self.agent_state = copy.deepcopy(self.agent_start_state)

		''' set other parameters '''
		self.restart_once_done = False  # restart or not once done
		self.verbose = True # to show the environment or not

		GridworldEnv.num_env += 1
		self.this_fig_num = GridworldEnv.num_env 
		# if self.verbose == True:
		#     self.fig = plt.figure(self.this_fig_num)
		#     plt.show(block=False)
		#     plt.axis('off')
		#     self._render()

	def _step(self, action):
		''' return next observation, reward, finished, success '''
		action = int(action)

		info = {}
		info['success'] = False
		nxt_agent_state = [self.agent_state[0] + self.action_pos_dict[action][0],
		                    self.agent_state[1] + self.action_pos_dict[action][1]]


		# if action == 0: # stay in place
		#     info['success'] = True
		#     return (self.agent_state, self.step_reward , False, info) # (self.observation, 0, False, info) 

		# Handle impossible behavior
		if nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]:
		    info['success'] = False
		    return (self.agent_state, self.out_of_grid  , False, info) # self.obs
		if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
		    info['success'] = False
		    return (self.agent_state, self.out_of_grid , False, info) # self.obs

		# Possible behavior
		org_color = self.current_grid_map[self.agent_state[0], self.agent_state[1]]
		new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
		if new_color == 0: # Background
			if org_color == 4:
			    self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
			    self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
			elif  org_color == 5 or org_color == 6 or org_color == 7:
			    self.current_grid_map[self.agent_state[0], self.agent_state[1]] = org_color-4 
			    self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
			self.agent_state = copy.deepcopy(nxt_agent_state)
		elif new_color == 1: # Lava
		   info['success'] = True #False
		   self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
		   self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
		   self.agent_state = copy.deepcopy(nxt_agent_state)
		   self.observation = self._gridmap_to_observation(self.current_grid_map)

		   return (self.agent_state, self.lava_reward, False, info) #self.observation
		elif new_color == 2 or new_color == 3: # If we reach a goal, change the color to show we reached it
		    self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
		    self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color+4
		    self.agent_state = copy.deepcopy(nxt_agent_state)
		    #print(nxt_agent_state, self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]])
		self.observation = self._gridmap_to_observation(self.current_grid_map)
		#self._render()
		if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1] : # Goal reached
		    target_observation = copy.deepcopy(self.observation)
		    if self.restart_once_done: # not really useful
		        self.agent_state = self.reset() #self.obs
		        info['success'] = True
		        return (self.agent_state, self.goal_reward, True, info)
		    else: 
		        info['success'] = True
		        return (self.agent_state, self.goal_reward, True, info) #target_observation
		else: # Not goal
		    info['success'] = True
		    return (self.agent_state, self.step_reward , False, info) #self.observation

	def step(self, action):
		if len(self.randomized_params) > 0:
			state, rew, done, info = self._step(action)
			additional_params = []
			if self.randomize :
				#self.randomized_params = list(self.np_random.choice(self.randomization_space, len(self.randomized_params)))
				additional_params = self.randomized_params
				info['state_randomized'] = copy.deepcopy(state) + additional_params
				if self.regularize :
					additional_params = self.regularized_params
			state = state + additional_params
			self.current_step += 1
			if self.current_step > self.max_episode_steps:
				done = True
			info['current_step'] = self.current_step
			#print(state)
			return (state, rew, done, info)
		else:
			state, rew, done, info = self._step(action)
			self.current_step += 1
			if self.current_step > self.max_episode_steps:
				done = True
			info['current_step'] = self.current_step
			return (state, rew, done, info)

	def _reset(self):
	    self.agent_state = copy.deepcopy(self.agent_start_state)
	    self.current_grid_map = copy.deepcopy(self.start_grid_map)
	    self.observation = self._gridmap_to_observation(self.start_grid_map)
	    self.current_step = 0
	    #self._render()
	    return self.agent_state #self.observation

	def reset(self):
		if len(self.randomized_params) > 0:
			state = self._reset()
			state_rand = copy.deepcopy(state)
			additional_params = []
			if self.randomize :
				self.randomized_params = list(self.np_random.choice(self.randomization_space, len(self.randomized_params)))
				additional_params = self.randomized_params
				state_rand = state_rand + additional_params
				if self.regularize :
					additional_params = self.regularized_params
			state = copy.deepcopy(state) + additional_params
			return state, state_rand
		else:
			return self._reset(), _

	def _read_grid_map(self, grid_map_path):
	    with open(grid_map_path, 'r') as f:
	        grid_map = f.readlines()
	    grid_map_array = np.array(
	        list(map(
	            lambda x: list(map(
	                lambda y: int(y),
	                x.split(' ')
	            )),
	            grid_map
	        ))
	    )
	    return grid_map_array

	def _get_agent_start_target_state(self, start_grid_map):
	    start_state = None
	    target_state = None
	    start_state = list(map(
	        lambda x:x[0] if len(x) > 0 else None,
	        np.where(start_grid_map == 4)
	    ))
	    target_state = list(map(
	        lambda x:x[0] if len(x) > 0 else None,
	        np.where(start_grid_map == 3)
	    ))
	    if start_state == [None, None] or target_state == [None, None]:
	        sys.exit('Start or target state not specified')
	    return start_state, target_state

	def _gridmap_to_observation(self, grid_map, obs_shape=None):
	    if obs_shape is None:
	        obs_shape = self.obs_shape
	    observation = np.zeros(obs_shape, dtype=np.float32)
	    gs0 = int(observation.shape[0]/grid_map.shape[0])
	    gs1 = int(observation.shape[1]/grid_map.shape[1])
	    for i in range(grid_map.shape[0]):
	        for j in range(grid_map.shape[1]):
	            observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1] = np.array(COLORS[grid_map[i,j]])
	    return observation

	def _render(self, mode='human', close=False):
		if self.verbose == False:
		    return
		img = self.observation
		fig = plt.figure(self.this_fig_num)
		plt.clf()
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img)
		title = 'No Randomization'
		if(len(self.randomized_params) > 0):
			plt.title('randomized_params : '+ str([round(rp,2) for rp in self.randomized_params]) )#+ ' regularized_params : '+ str(self.regularized_params) )
		fig.canvas.draw()
		plt.pause(0.00001) # 0.00001
		return 

	def render(self, mode='human', close=False):
		return self._render(mode,close)

	def seed(self, seed=1337):
		self.np_random, _ = seeding.np_random(seed)
		return [seed]
