import matplotlib.pyplot as plt
import numpy as np
from ica import *

SIGNALS = 2
SAMPLINGS = 1000

S = np.array([ const_powerd_samples(i+2, np.pi/(i+6), SAMPLINGS) for i in range(SIGNALS)]) 

# S = []
# for s in SC:
# 	S.append(s.real)
# 	S.append(s.imag)
# S = np.array(S)

# mean = np.mean(S,axis=1)
# S = S - np.array([np.full(SAMPLINGS, ave) for ave in mean ])

A = random_matrix(SIGNALS)

X = A @ S

# print(correlation(S))

r_res = FastICA(X.real, _assert=False)
i_res = FastICA(X.imag, _assert=False)

r_P = simple_circulant_P(A, r_res.W)
i_P = simple_circulant_P(A, i_res.W)

Y = r_P.T @ r_res.Y + i_P.T @ i_res.Y * 1j

plt.scatter(Y[0].real, Y[0].imag)
plt.show()
