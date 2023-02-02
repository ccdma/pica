import numpy as np

k1 = 1
k2 = 2

p1, q1 = 5, 2
p2, q2 = 3, 2

_sum = 0
for i in range(p1*p2):
    theta1 = pow(q1, i, p1)/p1
    theta2 = pow(q2, i, p2)/p2
    _sum += np.exp(-2j*np.pi*(theta1+theta2)*(k1+k2))
