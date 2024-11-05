## tests of some activation functions and what they actually look like

import numpy as np
import matplotlib.pyplot as plt

# W = np.array([1,2,-3, 0.4, -1.1])
W = 10 * (np.random.rand(5) - 0.5)
W2 = 10 * (np.random.rand(5) - 0.5)
# v = np.array([3,-2,1.3,0.8, -2.3])
v = 10 * (np.random.rand(5) - 0.5)

relu = lambda t : np.maximum(0, t)
# quadratic activation function
# f = lambda x : np.dot(v, np.power(x * W + W2, 2))
print(W)
print(W2)
f = lambda x : np.dot(v, relu(x * W+W2))

t = np.linspace(0,6, 200)
y = np.zeros(t.shape)
for idx,val in enumerate(t):
    y[idx] = f(val)

fig = plt.figure()
plt.plot(t,y,'b-')

plt.show()