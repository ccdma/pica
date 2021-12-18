import matplotlib.pyplot as plt
import numpy as np
from ica import *

np.random.seed(1)

def odd_or_even(i: int):
    if i%2 == 0:
        return 1
    else:
        return -1

signals = 10
samplings = 2000

S = np.array([ const_powerd_samples(2, np.pi/(np.sqrt(2)+i), samplings) for i in range(signals)])

B = np.array([[ odd_or_even(np.random.randint(0, 10)) for i in range(samplings) ] for _ in range(signals)])

T = S * B

# print(correlation(T.imag))

A = random_matrix(signals)
# print(la.det(A))

X = A @ T + np.random.normal(0.0, 0, (signals, samplings))

rr = FastICA(X.real, _assert=False)
ri = FastICA(X.imag, _assert=False)

rP = simple_circulant_P(A, rr.W)
iP = simple_circulant_P(A, ri.W)
S2 = rP.T @ rr.Y + (iP.T @ ri.Y) * 1j

RB = np.sign(S2.real*S.real+S2.imag*S.imag)

cber = np.abs(RB - B).mean()/2
print(cber)

# s = S[0]
# for i in range(signals):
# 	s = S[i]
# 	plt.scatter(s.real, s.imag, s=1)
# 	plt.plot(s.real, s.imag, lw=0.1)
# # plt.scatter(S.real, S.imag, s=1)
# plt.show()