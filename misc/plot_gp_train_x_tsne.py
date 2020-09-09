'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-04-15 12:09:24
@LastEditTime: 2020-05-12 22:03:44
@Description: 
'''

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


data = np.load('./gp_train_x.npy')
z_tsne = TSNE(n_components=2, learning_rate=10).fit_transform(data)

mark_size_1 = 15
mark_size_2 = 3

plt.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.5)
#plt.xticks([-20, -10, 0, 10, 20], fontsize=fontsize)
#plt.yticks([-20, -10, 0, 10, 20], fontsize=fontsize)
plt.tight_layout()
frame = plt.gca()
#frame.axes.get_yaxis().set_visible(False)
#frame.axes.get_xaxis().set_visible(False)
#frame.spines['top'].set_visible(False) 
#frame.spines['bottom'].set_visible(False) 
#frame.spines['left'].set_visible(False) 
#frame.spines['right'].set_visible(False)

plt.show()
