import matplotlib.pyplot as plt
import numpy as np
from ica import *

SIGNALS = 2
SAMPLINGS = 1000

SC = np.array([ const_powerd_samples(i+2, np.pi/(i+6), SAMPLINGS) for i in range(SIGNALS)]) 

S = []
for s in SC:
	S.append(s.real)
	S.append(s.imag)
S = np.array(S)

mean = np.mean(S,axis=1)
S = S - np.array([np.full(SAMPLINGS, ave) for ave in mean ])

A = random_matrix(SIGNALS*2)

X = A @ S

print(correlation(S))

res = FastICA(X, _assert=False)

P = simple_circulant_P(A, res.W)
Y = res.Y
Y = P.T @ res.Y

plt.scatter(Y[0], Y[1])
plt.show()
