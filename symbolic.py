from numpy import random
from ica import *
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def odd_or_even(i: int):
    if i%2 == 0:
        return 1
    else:
        return -1

def cte(C: np.ndarray):
    size = C.shape[0]
    _results = []
    for i in range(size):
        absvec = np.abs(C[i])
        max_k = np.max(absvec)
        _sum = np.sum(absvec)
        _results.append(_sum/max_k-1)
    for i in range(size):
        absvec = np.abs(C[:, i])
        max_k = np.max(absvec)
        _sum = np.sum(absvec)
        _results.append(_sum/max_k-1)
    return sum(_results)

SIGNALS = 3
SAMPLINGS = 1000

B = np.array([[ odd_or_even(np.random.randint(0, 10)) for i in range(SAMPLINGS) ] for _ in range(SIGNALS)])
S = np.array([ chebyt_samples(2, 0.1+i/10, SAMPLINGS) for i in range(SIGNALS)])

T = S * B

print(correlation(T))

A = random_matrix(SIGNALS)

X = A @ T + np.random.normal(0.0, 0.1, (SIGNALS, SAMPLINGS))

res = FastICA(X, _assert=False)

Y = res.Y
P = simple_circulant_P(A, res.W)

S2 = P.T @ Y

RB = np.sign(S2*S)

# BER
print(np.abs(RB - B).mean()/2)
# CTE
print(cte(res.W @ A))