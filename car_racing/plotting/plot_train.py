import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

PATH = 'results/'
NUM_SEEDS = 5

normal_results = []
randomized_results = []
regularized_results = []


for dirs in os.listdir(PATH):

    info = eval(open(PATH + dirs + '/experimental-setup','r').read())

    if info['randomize']:
        agent_type = 'randomize'
    else:
        if info['lamda'] == 50:
            agent_type = 'regularize'
        elif info['lamda'] == 0:
            agent_type = 'normal'
        else:
            agent_type = 'na'

    if agent_type in ['randomize','normal','regularize']:

        log_data_filename = PATH + dirs + '/log_data.pkl'
        log_data = pickle.load(open(log_data_filename, 'rb'))
        score_data = np.array(log_data['Episode_score'])
        scores = score_data[:,0]
        episodes = score_data[:,1]/200

        if agent_type == 'normal':
            normal_results.append(scores)
        elif agent_type == 'regularize':
            regularized_results.append(scores)
        else:
            randomized_results.append(scores)

mean_normal = np.array(normal_results).mean(axis=0)
mean_normal = savgol_filter(mean_normal, 51, 3)
std_normal = savgol_filter(np.array(normal_results).std(axis=0), 51, 3)

mean_reg = np.array(regularized_results).mean(axis=0)
mean_reg = savgol_filter(mean_reg, 51, 3)
std_reg = savgol_filter(np.array(regularized_results).std(axis=0), 51, 3)

mean_rand = np.array(randomized_results).mean(axis=0)
mean_rand = savgol_filter(mean_rand, 51, 3)
std_rand = savgol_filter(np.array(randomized_results).std(axis=0), 51, 3)


plt.figure(figsize=(10,5))
plt.plot(episodes, mean_normal, label = 'Normal', color = 'chocolate')
plt.plot(episodes, mean_reg, label = 'Regularized', color = 'blue')
plt.plot(episodes, mean_rand, label = 'Randomized', color = 'green')

plt.fill_between(episodes,
               np.maximum(mean_normal - 1.96 * std_normal/np.sqrt(NUM_SEEDS),-100),
               mean_normal + 1.96 * std_normal/np.sqrt(NUM_SEEDS),
               facecolor='orange',
               alpha=0.2)

plt.fill_between(episodes,
               np.maximum(mean_reg - 1.96 * std_reg/np.sqrt(NUM_SEEDS),-100),
               mean_reg + 1.96 * std_reg/np.sqrt(NUM_SEEDS),
               facecolor='blue',
               alpha=0.2)

plt.fill_between(episodes,
               np.maximum(mean_rand - 1.96 * std_rand/np.sqrt(NUM_SEEDS),-100),
               mean_rand + 1.96 * std_rand/np.sqrt(NUM_SEEDS),
               facecolor='green',
               alpha=0.2)

plt.gcf().subplots_adjust(bottom=0.2)
plt.title('Training curves', fontsize = 20)
plt.xlabel('Episode', fontsize = 18)
plt.ylabel('Score', fontsize = 18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize = 18)
plt.show()