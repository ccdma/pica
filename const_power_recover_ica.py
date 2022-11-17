"""
ICAを用いてパワー一定カオス拡散符号の復元を行う
"""
import matplotlib.pyplot as plt
import numpy as np
import lb

np.random.seed(1)

SIGNALS = 2
SAMPLINGS = 300

S = np.array([ lb.const_power_code(2, np.pi/(np.sqrt(2)+i), SAMPLINGS) for i in range(SIGNALS)]) 

# S = []
# for s in SC:
# 	S.append(s.real)
# 	S.append(s.imag)
# S = np.array(S)

# mean = np.mean(S,axis=1)
# S = S - np.array([np.full(SAMPLINGS, ave) for ave in mean ])

A = lb.random_matrix(SIGNALS)

X = A @ S

print(lb.correlation(np.vstack([S.real, S.imag])))

r_res = lb.fast_ica(X.real, _assert=False)
i_res = lb.fast_ica(X.imag, _assert=False)

r_P = lb.simple_circulant_P(A, r_res.W)
i_P = lb.simple_circulant_P(A, i_res.W)

Y = r_P.T @ r_res.Y + i_P.T @ i_res.Y * 1j

Y = Y/np.abs(Y)

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

ax[0].set_title("source", fontsize=16)
ax[1].set_title("mixed", fontsize=16)
ax[2].set_title("reconstruct", fontsize=16)
for a in ax:
	a.set_xlabel("real", fontsize=14)
	a.set_ylabel("image", fontsize=14)
	a.tick_params(labelsize=10)

# fig.tight_layout()
# fig.suptitle("", x=0.1, y=0.97)
fig.set_figheight(5)
fig.set_figwidth(17)

plt.show()
