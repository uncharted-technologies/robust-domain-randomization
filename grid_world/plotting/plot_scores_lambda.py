"""
This code loads the results obtained from the calculate_scores_lambda script and plots them, producing figure 2 in our paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as colors


R_MAX = 1 #Max reward in the MDP
N_POINTS = 20 #Number of seeds to plot, per value of lambda

scores_train = np.load('results/reg_scores_train_vpg.npy')
scores = np.load('results/reg_scores_vpg.npy')
lipschitz = np.load('results/reg_lipschitz_vpg.npy')
lamda_range = np.geomspace(0.00000001,0.1,30)

empirical_difference = np.abs(scores_train-np.mean(scores,axis=2))
theory_bound = 2 * R_MAX * np.sum(np.array([np.minimum(1,(1+t)*lipschitz) for t in np.arange(10)]),axis=0)

colors = cm.rainbow(np.linspace(0, 1, len(lamda_range)))

theory_bound_reshaped = theory_bound[:,:N_POINTS].flatten()
empirical_difference_reshaped = empirical_difference[:,:N_POINTS].flatten()
condition = np.repeat(lamda_range[:,np.newaxis],N_POINTS,axis=1)
condition = np.log10(condition.flatten())


plt.scatter(theory_bound_reshaped,empirical_difference_reshaped,c=condition,cmap='rainbow')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')

cbar = plt.colorbar(ticks=[-8, -6, -4, - 2])
cbar.ax.set_yticklabels([r'$10^{-8}$',r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$'],fontsize = 22)
cbar.set_label(r'$\lambda$', rotation=0,fontsize = 26)

plt.title('Generalization bound', fontsize = 30)
plt.xlabel('Bound on ' + r'$|\eta_1-\eta_2|$', fontsize = 26)
plt.ylabel('Experimental ' + r'$|\eta_1-\eta_2|$', fontsize = 26)
plt.gcf().subplots_adjust(left=0.2,bottom=0.18)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)

plt.plot(np.linspace(0.01,11,1000),np.linspace(0.01,11,1000), '--', color = 'k')


plt.show()
