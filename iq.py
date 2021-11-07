from pica.ica import *
import matplotlib.pyplot as plt
import numpy as np

def odd_or_even(i: int):
    if i%2 == 0:
        return 1
    else:
        return -1

SIGNALS = 3
SAMPLINGS = 1000

bits = np.array([[ odd_or_even(np.random.randint(0, 10)) for i in range(SAMPLINGS) ] for _ in range(SIGNALS)])
S = np.array([ chebyt_samples(i+2, 0.1, SAMPLINGS) for i in range(SIGNALS)])

T = S * bits

print(correlation(T))

A = random_matrix(SIGNALS)

X = A @ T

res = FastICA(X, _assert=False)

Y = res.Y

plt.scatter(Y[1][:-1], Y[1][1:])
plt.show()