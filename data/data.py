'''
@Author: Mengdi Xu
@Email: 
@Date: 2020-03-17 12:32:22
@LastEditTime: 2020-03-17 20:10:02
@Description: 
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# random data generation
num = 20
slope = np.linspace(0, 5, 3)
# x = np.random.uniform(0,10,100)
x = np.linspace(0, 5, num)
a = np.random.uniform(1, 2, num)
data = []
count = 0
for s in slope:
    y = x + s*a
    count += 1
    data_s = np.vstack((np.ones(num)*count, x, a, y)).T
    data.extend(data_s)

data = np.array(data)
np.save('data_sparse', data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(5):
    start = i*num
    x = data[start:start+num, 1]
    a = data[start:start+num, 2]
    x_n = data[start:start+num, 3]
    ax.scatter(x, a, x_n)

ax.set_xlabel('x')
ax.set_ylabel('a')
ax.set_zlabel('x_n')

#plt.savefig('data_vis')
plt.show()
plt.close()
