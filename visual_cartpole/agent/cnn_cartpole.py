
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
        
class CNN_cartpole(nn.Module):

	def __init__(self, observation_space, n_outputs, add_current_step,width=256):
	    # CNN architechture of DeepMind described in https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf :

	    # The first hidden layer convolves 32 filters of 8 3 8 with stride 4 with the input image and applies a rectifier nonlinearity
	    # The second hidden layer convolves 64 filters of 4 3 4 with stride 2, again followed by a rectifier nonlinearity
	    # This is followed by a third convolutional layer that convolves 64 filters of 3 3 3 with stride 1 followed by a rectifier.
	    # The final hidden layer is fully-connected and consists of 512 rectifier units.

	    super().__init__()

	    if len(observation_space.shape) != 3:
	        raise NotImplementedError

	    self.add_current_step = add_current_step

	    n_intermediate = 3137 if self.add_current_step else 3136
	    # Defining the network architechture
	    self.conv = nn.Sequential(nn.Conv2d(9, 32, 8, stride=4), # 32 , 8
	                              nn.ReLU(),
	                              nn.Conv2d(32, 64, 4, stride=2),
	                              nn.ReLU(),
	                              nn.Conv2d(64, 64, 3, stride=1),
	                              nn.ReLU())

	    self.output = nn.Sequential(nn.Linear(n_intermediate, width), #3136
	                              nn.ReLU(),
	                              nn.Linear(width, n_outputs))

	    self.conv.apply(lambda x: init_weights(x, np.sqrt(2)))
	    self.output.apply(lambda x: init_weights(x, np.sqrt(2)))

	    
	def forward(self, obs, current_step):
		if len(obs.shape) != 4:
			obs = obs.unsqueeze(0)
		obs = obs.permute(0,3,1,2)
		obs = obs/255
		obs = self.conv(obs)
		#current_step = current_step/200
		obs = obs.view(obs.size(0), -1)


		if self.add_current_step:
			if len(current_step.shape) == 1:
				current_step = current_step.unsqueeze(1)
			obs = torch.cat((obs,current_step),dim=1)

		return self.output(obs)

	def get_features(self, obs, current_step):
		"""
		Gets activations from last hidden layer, to be used in our regularization scheme
		"""
		if len(obs.shape) != 4:
			obs = obs.unsqueeze(0)
		obs = obs.permute(0,3,1,2)
		obs = obs/255
		obs = self.conv(obs)
		obs = obs.view(obs.size(0), -1)

		
		if self.add_current_step : 
			if len(current_step.shape) == 1:
				current_step = current_step.unsqueeze(1)

			obs = torch.cat((obs,current_step),dim=1)

		obs = self.output[:-1](obs)
		return obs