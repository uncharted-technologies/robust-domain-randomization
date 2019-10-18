import sys
sys.path.append('../')

import numpy as np
from env import gym
import matplotlib.pyplot as plt
import cv2

cv2.ocl.setUseOpenCL(False)


from threading import Event, Thread
import pickle
from collections import deque
from gym import spaces

import random


class CartPole_Pixel(gym.Wrapper):
	"""
	Wrapper for getting raw pixel in cartpole env
	observation: 400x400x1 => (Width, Height, Colour-chennel)
	we dispose 100pxl from each side of width to make the frame divisible(Square) in CNN
	"""
	def __init__(self, env, randomize, regularize, reference_domain ,colors):
		self.width  = 84
		self.height = 84

		gym.Wrapper.__init__(self, env)
		self.env = env#env.unwrapped

		#self.env.seed(123)  # fix the randomness for reproducibility purpose

		#self.observation_space = np.zeros((self.width, self.height, 4))
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
		#print(self.env.observation_space)
		self.randomize = randomize
		self.regularize = regularize
		self.sampled_color = reference_domain
		self.reference_domain = reference_domain

		"""
		start new thread to deal with getting raw image
		"""
		#from tf_rl.env.cartpole_pixel import RenderThread
		self.renderer = RenderThread(env)
		self.renderer.start()

		self.invariant_states = [[p,0.,0.,0.] for p in np.linspace(-2.4, 2.4, 15)]
		self.colors = colors
		self.current_step = 0

	def _pre_process(self, frame):
		#print(frame.shape)
		#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
		
		#print(frame)
		#frame = np.ascontiguousarray(frame)#(frame.transpose(2,0,1)[np.newaxis,:])
		#if self.randomize and reset:
			#self.env.viewer.set_color(0.,0.,0.)
		#frame = np.expand_dims(frame, -1)
		#print(frame.shape)
		return frame

	def step(self, ac):
		#self.renderer.background_color = None
		self.renderer.background_color = self.reference_domain 
		_, reward, done, info = self.env.step(ac)
		self.current_step += 1
		info['current_step'] = self.current_step
		#print(reward)
		self.renderer.background_color = self.reference_domain 
		if self.randomize:
			self.renderer.background_color = self.sampled_color
			if self.regularize:
				self.renderer.background_color = self.reference_domain
		self.renderer.begin_render() # move screen one step
		self.observation = self._pre_process(self.renderer.get_screen())

		if done:
			reward = -1.0  # reward at a terminal state
		return self.observation, reward, done, info

	def reset(self, **kwargs):
		self.renderer.background_color = self.reference_domain 
		if self.randomize:
			self.sampled_color =  random.choice(self.colors)
			self.renderer.background_color = self.sampled_color
			if self.regularize :
				self.renderer.background_color = self.reference_domain
		self.env.reset()
		self.current_step = 0
		self.renderer.begin_render() # move screen one step
		self.observation = self._pre_process(self.renderer.get_screen())
		return  self.observation# overwrite observation by raw image pixels of screen

	def close(self):
		self.renderer.stop() # terminate the threads
		self.renderer.join() # collect the dead threads and notice all threads are safely terminated
		if self.env:
			return self.env.close()

	def get_selected_features(self, k=4):
		tmp = []

		for state in self.invariant_states :
			self.env.state = state
			self.renderer.begin_render()
			observation = self._pre_process(self.renderer.get_screen())
			#stacked_observation = []
			#for _ in range(k):
			#	stacked_observation.append(observation)
			tmp.append(observation)


		return tmp

	#def set_background_color(self, background_color):
	#	self.current_background_color = background_color
	#	self.renderer.background_color = self.current_background_color


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
	def __init__(self, env, k, get_all_randomized_per_step = False):
		"""Stack k last frames.
		Returns lazy array, which is much more memory efficient.
		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		self.frames_randomized = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

		# For t-SNE plots and value function std, get all randomized states per step (and not only current sampled color)
		self.get_all_randomized_per_step = get_all_randomized_per_step
		if self.get_all_randomized_per_step :
			self.frames_randomized = []
			for i in range(len(self.env.colors)):
				self.frames_randomized.append(deque([], maxlen=k))



	def reset(self):
		ob = self.env.reset()
		if self.get_all_randomized_per_step:
			for _ in range(self.k):
				self.frames.append(ob)

			for i, col in enumerate(self.env.colors):
				self.env.sampled_color = col
				rand_ob = self.randomize_observation(ob)
				for _ in range(self.k):
					self.frames_randomized[i].append(rand_ob)
		else : 
		    rand_ob = self.randomize_observation(ob)
		    for _ in range(self.k):
		        self.frames.append(ob)
		        self.frames_randomized.append(rand_ob)
		return self._get_ob()

	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)

		if self.get_all_randomized_per_step:
			for i, col in enumerate(self.env.colors):
				self.env.sampled_color = col
				self.frames_randomized[i].append(self.randomize_observation(ob))
		else :
		    self.frames_randomized.append(self.randomize_observation(ob))

		state,state_randomized = self._get_ob()
		info['state_randomized'] = state_randomized
		return state, reward, done, info

	def _get_ob(self):
		if self.get_all_randomized_per_step:
			assert len(self.frames) == self.k and all([len(fr) == self.k for fr in self.frames_randomized])
			return LazyFrames(list(self.frames)), [LazyFrames(list(fr)) for fr in self.frames_randomized]
		else :
		    assert len(self.frames) == self.k and len(self.frames_randomized) == self.k
		    return LazyFrames(list(self.frames)), LazyFrames(list(self.frames_randomized))

	def get_stacked(self,observation):
		for _ in range(self.k):
		    self.frames.append(observation)
		return self._get_ob()

	def randomize_observation(self, observation):
		rand_ob = observation.copy()
		# TODO must change rand_ob == 255 to the reference domain ( in this case it was always white )
		rand_ob[np.all(rand_ob == 255., axis = 2)] = (self.env.sampled_color[0] * 255., self.env.sampled_color[1] * 255., self.env.sampled_color[2] * 255.)
		return rand_ob


class RenderThread(Thread):
	"""
	Original Code:
		https://github.com/tqjxlm/Simple-DQN-Pytorch/blob/master/Pytorch-DQN-CartPole-Raw-Pixels.ipynb
	Data:
		- Observation: 3 x 400 x 600
	Usage:
		1. call env.step() or env.reset() to update env state
		2. call begin_render() to schedule a rendering task (non-blocking)
		3. call get_screen() to get the lastest scheduled result (block main thread if rendering not done)
	Sample Code:
	```python
		# A simple test
		env = gym.make('CartPole-v0').unwrapped
		renderer = RenderThread(env)
		renderer.start()
		env.reset()
		renderer.begin_render()
		for i in range(100):
			screen = renderer.get_screen() # Render the screen
			env.step(env.action_space.sample()) # Select and perform an action
			renderer.begin_render()
			print(screen)
			print(screen.shape)
		renderer.stop()
		renderer.join()
		env.close()
	```
	"""

	def __init__(self, env):
		super(RenderThread, self).__init__(target=self.render)
		self._stop_event = Event()
		self._state_event = Event()
		self._render_event = Event()
		self.env = env
		self.background_color = None

	def stop(self):
		"""
		Stops the threads
		:return:
		"""
		self._stop_event.set()
		self._state_event.set()

	def stopped(self):
		"""
		Check if the thread has been stopped
		:return:
		"""
		return self._stop_event.is_set()

	def begin_render(self):
		"""
		Start rendering the screen
		:return:
		"""
		self._state_event.set()

	def get_screen(self):
		"""
		get and output the screen image
		:return:
		"""
		self._render_event.wait()
		self._render_event.clear()
		return self.screen

	def render(self):
		while not self.stopped():
			self._state_event.wait()
			self._state_event.clear()
			self.screen = self.env.render(mode='rgb_array', background_color=self.background_color)
			self._render_event.set()