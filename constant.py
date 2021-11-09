import matplotlib.pyplot as plt
import numpy as np
from ica import *

np.random.seed(1)

SIGNALS = 2
SAMPLINGS = 300

S = np.array([ const_powerd_samples(2, np.pi/(i+11), SAMPLINGS) for i in range(SIGNALS)]) 

# S = []
# for s in SC:
# 	S.append(s.real)
# 	S.append(s.imag)
# S = np.array(S)

# mean = np.mean(S,axis=1)
# S = S - np.array([np.full(SAMPLINGS, ave) for ave in mean ])

A = random_matrix(SIGNALS)

X = A @ S

print(correlation(np.vstack([S.real, S.imag])))

r_res = FastICA(X.real, _assert=False)
i_res = FastICA(X.imag, _assert=False)

r_P = simple_circulant_P(A, r_res.W)
i_P = simple_circulant_P(A, i_res.W)

Y = r_P.T @ r_res.Y + i_P.T @ i_res.Y * 1j

fig, ax = plt.subplots(1, 3)

r_size = 6
lw = 0.2
for i in range(SIGNALS):
	s = S[i]
	ax[0].scatter(s.real, s.imag, alpha=0.5, s=r_size)
	ax[0].plot(s.real, s.imag, lw=lw)
for i in range(SIGNALS):
	x = X[i]
	ax[1].scatter(x.real, x.imag, alpha=0.5, s=r_size)
	ax[1].plot(x.real, x.imag, lw=lw)
for i in range(SIGNALS):
	y = Y[i]
	ax[2].scatter(y.real, y.imag, alpha=0.5, s=r_size)
	ax[2].plot(y.real, y.imag, lw=lw)

ax[0].set_title("source")
ax[1].set_title("mixed")
ax[2].set_title("reconstruct")
for a in ax:
	a.set_xlabel("real")
	a.set_ylabel("image")

# fig.tight_layout()
# fig.suptitle("", x=0.1, y=0.97)
fig.set_figheight(5)
fig.set_figwidth(16)

plt.show()
