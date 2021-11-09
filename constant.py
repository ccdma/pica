import matplotlib.pyplot as plt
import numpy as np
from ica import *

np.random.seed(0)

SIGNALS = 2
SAMPLINGS = 300

S = np.array([ const_powerd_samples(2, np.pi/(i+6), SAMPLINGS) for i in range(SIGNALS)]) 

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

fig, ax = plt.subplots(1, 3)
for i in range(SIGNALS):
	ax[0].scatter(S[i].real, S[i].imag, alpha=0.5)
for i in range(SIGNALS):
	ax[1].scatter(X[i].real, X[i].imag, alpha=0.5)
for i in range(SIGNALS):
	ax[2].scatter(Y[i].real, Y[i].imag, alpha=0.5)
ax[0].set_title("source")
ax[1].set_title("mixed")
ax[2].set_title("reconstruct")

fig.tight_layout()
fig.suptitle("", x=0.1, y=0.97)
fig.set_figheight(5)
fig.set_figwidth(12)

plt.show()
