'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-25 09:49:07
@LastEditTime: 2020-03-25 12:45:49
@Description:
'''

import numpy as np 
import torch
import timeit

setup1 = """
import numpy as np 
import torch
def test1():
	a,b = -2,2
	y = (b-a)*np.random.rand(128, 500000)+a

"""

setup1_1 = """
import numpy as np 
import torch
def test1_1():
	y = np.random.uniform(-2, 2, [128, 500000])

"""



setup2 = """
import numpy as np 
import torch
def test2():
	a,b = -2,2
	y = (b-a)*torch.rand(128, 500000)+a
	y = y.numpy()
"""

setup3 = """
import numpy as np 
import torch
uniform = torch.distributions.uniform.Uniform
uni = uniform(-2, 2)
def test3():
	r = uni.sample((128,500000))
	r = r.numpy()
	#print(r.shape)
"""

# GPU sampling
setup4 = """
import numpy as np 
import torch

def test4():

	y = torch.empty((128,500000)).cuda()
	# print(y.shape)
	y.uniform_(-2,2)
	x = y.cpu().numpy()
	#print(x.shape,x.min(),x.max(),x.mean())
"""



setup5 = """
import numpy as np 
import torch
shape = (128,500000)
uniform = torch.distributions.uniform.Uniform
def test5():
	y = torch.empty(shape).cuda()
	torch.rand(y.size(), out=y)
	x = y.cpu().numpy()
	#print(x.shape,x.min(),x.max(),x.mean())
"""

if __name__ == "__main__":
	n = 1
	#print("test1: ", timeit.timeit("test1()", setup=setup1, number=n))
	#print("test1_1: ", timeit.timeit("test1_1()", setup=setup1_1, number=n))
	print("test2: ", timeit.timeit("test2()", setup=setup2, number=n))
	print("test3: ", timeit.timeit("test3()", setup=setup3, number=n))
	print("test4: ", timeit.timeit("test4()", setup=setup4, number=n))
	print("test5: ", timeit.timeit("test5()", setup=setup5, number=n))
	print("test4: ", timeit.timeit("test4()", setup=setup4, number=n))
