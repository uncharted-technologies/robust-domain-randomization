"""
Loads results from generalization_test in the result folder and plots them,
yielding figure 4

"""


import matplotlib.pyplot as plt
import numpy as np

reds = np.linspace(0., 1., 6)
blues = np.linspace(0., 1., 6)
xx, yy = np.meshgrid(reds, blues)

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1,aspect='equal')
plt.gcf().subplots_adjust(bottom=0.2)


scores_reg = np.load('results/interpolation_green_reg_rand.npy')[0] 
plt.pcolormesh(xx.T, yy.T, np.array(scores_reg).reshape(len(reds)-1,len(blues)-1), vmin= 35., vmax = 185., cmap = plt.get_cmap('PiYG'))#, shading='gouraud')
plt.title('Scores - Regularized', fontsize = 24)
plt.xlabel('Red', fontsize = 24)
plt.ylabel('Blue', fontsize = 24)
plt.xticks([0,0.5,1],fontsize=24)
plt.yticks(fontsize=24)

plt.subplot(1, 3, 2,aspect='equal')
scores_rand = np.load('results/interpolation_green_reg_rand.npy')[1]
plt.pcolormesh(xx.T, yy.T, np.array(scores_rand).reshape(len(reds)-1,len(blues)-1), vmin= 35., vmax = 185.,cmap = plt.get_cmap('PiYG'))
#cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=24)
plt.title('Scores - Randomized', fontsize = 24)
plt.xlabel('Red', fontsize = 24)
#plt.ylabel('Blue', fontsize = 24)
plt.xticks([0,0.5,1],fontsize=24)
plt.yticks([])
#plt.yticks(fontsize=24)

plt.subplot(1, 3, 3)
scores_rand = np.load('results/interpolation_green_normal.npy')[2][2]
plt.pcolormesh(xx.T, yy.T, np.array(scores_rand).reshape(len(reds)-1,len(blues)-1), vmin= 35., vmax = 185.,cmap = plt.get_cmap('PiYG'))
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=24)
plt.title('Scores - Normal', fontsize = 24)
plt.xlabel('Red', fontsize = 24)
#plt.ylabel('Blue', fontsize = 24)
plt.xticks([0,0.5,1],fontsize=24)
plt.yticks([])
#plt.yticks(fontsize=24)


plt.show()


