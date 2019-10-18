"""
Loads tsnes calculated from tsne_features and saved in the results folder and plots them.
Yields figure 5 in our paper.
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Only need to modify these 2 lines to get all t-SNEs
to_plot = 'color' # Choose here which tsne to plot, 'color' or 'gray' ('gray' not implemented here)
name = 'Randomized' # 'Regularized' or 'Randomized'


prefix_tsne = 'TSNE_onTrain_'
prefix_label = 'labelsTSNE_onTrain_'

color_range = np.linspace(0.7, 1., 3)
color_range2 = np.linspace(0., 0.3, 3)
colors = [(r,g,b) for r in color_range for g in color_range for b in color_range ] +  [(r,g,b) for r in color_range2 for g in color_range2 for b in color_range2 ]

colors = [(1,0.5,1),[0,1,0]]

if to_plot == 'gray':
    prefix_tsne = 'TSNE_'
    prefix_label = 'labelsTSNE_'
    colors = [(x,x,x) for x in reversed(np.linspace(0,1,11))]

se = 0
X_embedded = np.load('results/'+str(prefix_tsne)+ str(name) + '_'+str(se) + '.npy')
labels = np.load('results/'+ str(prefix_label)+ str(name) + '_'+str(se) + '.npy')
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels,cmap=matplotlib.colors.ListedColormap(colors),linewidths=2)
plt.title(str(name),fontsize = 30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()