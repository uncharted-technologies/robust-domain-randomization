"""
Loads networks for trained agents and plots training curves shown in figure 3.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


NUM_SEEDS = 10

def plot_mean_and_confidence_intervals(x, results_array):
    plt.figure(figsize=(10,5))


    mean_normal = savgol_filter(results_array[0].mean(axis=0),51,3)
    std_normal = savgol_filter(results_array[0].std(axis=0),51,3)
    mean_reg = savgol_filter(results_array[1].mean(axis=0),51,3)
    std_reg = savgol_filter(results_array[1].std(axis=0),51,3)
    mean_rand = savgol_filter(results_array[2].mean(axis=0),51,3)
    std_rand = savgol_filter(results_array[2].std(axis=0),51,3)

    plt.plot(x, mean_normal, label = 'Normal', color = 'chocolate')
    plt.plot(x, mean_reg, label = 'Regularized', color = 'blue')
    plt.plot(x, mean_rand, label = 'Randomized', color = 'green')
    plt.fill_between(x,
               np.maximum(mean_normal - 1.96* std_normal/np.sqrt(NUM_SEEDS),0),
               np.minimum(mean_normal + 1.96* std_normal/np.sqrt(NUM_SEEDS),200.),
               facecolor='orange',
               alpha=0.2)
    plt.fill_between(x,
                np.maximum(mean_reg - 1.96* std_reg/np.sqrt(NUM_SEEDS),0),
                np.minimum(mean_reg+ 1.96* std_reg/np.sqrt(NUM_SEEDS),200.),
                facecolor='blue',
                alpha=0.2)
    plt.fill_between(x,
                np.maximum(mean_rand - 1.96* std_rand/np.sqrt(NUM_SEEDS),0),
                np.minimum(mean_rand + 1.96* std_rand/np.sqrt(NUM_SEEDS),200.),
                facecolor='green',
                alpha=0.2)


TYPE = 'big_lam10' # small or big_lam10 or big_lam100 (lam10 used in paper)
N_SEEDS =  10 
nb_steps = 100000
NUM_DISCRETIZE = 200
MAX_EPISODE_LENGTH = 200
xtrain = np.linspace(MAX_EPISODE_LENGTH + 50, nb_steps-MAX_EPISODE_LENGTH-50, num=NUM_DISCRETIZE, endpoint=True) # training

training_curves = np.zeros((3,N_SEEDS,NUM_DISCRETIZE))

names = ['Normal_', 'Regularized_', 'Randomized_' ]

for k,name in enumerate(names):
    for se in range(N_SEEDS):
        typeOf = 'results/training_curves/' + str(TYPE) + '/' + name  +str(se) + '/'
        logdata = pickle.load(open(typeOf + "log_data.pkl",'rb'))
        scores = np.array(logdata['Episode_score'])
        ftrain = interp1d(scores[:, 1], scores[:, 0], kind='nearest')
        print(name,min(scores[:, 1]), max(scores[:, 1]))
        training_curves[k,se] = ftrain(xtrain)

plot_mean_and_confidence_intervals(xtrain, training_curves)
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('Training curves, Big Domain', fontsize = 20)
plt.xlabel('Step', fontsize = 18)
plt.ylabel('Score', fontsize = 18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize = 18)
plt.show()
