"""
ICAによる系列の再構成
"""
import lb
import matplotlib.pyplot as plt
import numpy as np

lb.set_seed(4)

"""
	最小値=-1,最大値=1で正規化する
"""
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result01 = (x-min)/(max-min)
    return result01*2-1

LENGTH = 1000

sources = np.vstack([
	[lb.chebyt_code(2, a0, LENGTH) for a0 in [0.1, 0.2,]],
	[lb.chebyt_code(3, a0, LENGTH) for a0 in [0.1, 0.2,]]
])

USERS = sources.shape[0]

A = lb.random_matrix(USERS)

mixed = A @ sources

fast_ica_res = lb.fast_ica(mixed)
P = lb.estimate_circulant_matrix(A, fast_ica_res.W)
reconstruct = P.T @ fast_ica_res.Y

mixed = min_max(mixed, axis=1)
reconstruct = min_max(reconstruct, axis=1)

# return-mapのプロット
nrows = 1
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5,nrows*5))
# fig.gca().set_aspect('equal','datalim')

for code in sources:
	ax = axes[0]
	ax.scatter(code[:-1], code[1:])
	ax.set_title("sources")
for code in mixed:
	ax = axes[1]
	ax.scatter(code[:-1], code[1:])
	ax.set_title("mixed")
for code in reconstruct:
	ax = axes[2]
	ax.scatter(code[:-1], code[1:])
	ax.set_title("reconstruct")

fig.tight_layout()
plt.show()

