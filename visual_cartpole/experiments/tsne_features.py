
"""
Loads trained agents from the results folder and calculates tsnes of their last hidden layer
for all states encountered during a single greedy rollout. Results are saved in the results folder
and are plotted with plot_tsne in the plotting folder.
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

from sklearn.manifold import TSNE
import matplotlib
import torch


ADD_CURRENT_STEP = True
reference_domain = (1.,1.,1.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NUM_SEEDS = 1

SAVE  = True # Save tsnes as npy ? 

# Choose colors to calculate features on

colors = [(1,0.5,1),[0,1,0]]

# Paths to agents networks
PATH_PREFIX = 'results/networks/all_networks/big_lam10/'
agents = {#'Normal' : [PATH_PREFIX + 'Normal_'+str(i)+'/network.pth'  for i in range(NUM_SEEDS)], #No need for normal in this experiment
'Regularized' : [PATH_PREFIX + 'Regularized_'+str(i)+'/network.pth'for i in range(NUM_SEEDS)],
'Randomized' : [PATH_PREFIX + 'Randomized_'+str(i)+'/network.pth' for i in range(NUM_SEEDS)]}

env = gym.make("CartPole-v0")
env = CartPole_Pixel(env, False, False, reference_domain, colors) # Randomize = False, regularize = False for tsne plots
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



for name_id, name in enumerate(agents.keys()):
	for se in range(NUM_SEEDS):
		print('seed : ', se)
		features = []
		values = []
		labels = []

		agent.load(agents[name][se])

		obs, rand_obs = env.reset()
		returns = 0
		returns_list = []
		current_step = 0.
		for i in range(300): 
			for lab,rand_ob in enumerate(rand_obs):

				feat = agent.network.get_features(torch.FloatTensor(rand_ob).to(device), torch.FloatTensor([current_step]).to(device)).detach().cpu().numpy()[0]
				features.append(feat)
				
				values.append(agent.network(torch.FloatTensor(rand_ob).to(device), torch.FloatTensor([current_step]).to(device)).max().item())
				labels.append(lab)

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

		X_embedded = TSNE(n_components=2).fit_transform(np.array(features))
		if SAVE:
			np.save('TSNE_onTrain_' + str(name) + '_'+str(se) + '.npy',X_embedded)
			np.save('labelsTSNE_onTrain_' + str(name) + '_'+str(se) + '.npy',np.array(labels))
		plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels,cmap=matplotlib.colors.ListedColormap(colors))
		plt.title(str(name) + ' score : ' + str(np.mean(returns_list)))
		plt.show()

env.close()