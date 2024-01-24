"""
原始根^n符号の複数個の特性
"""
import lb, numba, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)
plt.rcParams['font.size'] = 15

"""
自己相関を計算（冒頭の1を含まない）
"""
def self_correlations(code):
	return lb.cross_correlations(code, code)[1:]

pq_set = [(13,2), (19,2)]

code_len = lb.mixed_primitive_root_code(pq_set, 1).shape[0]

k_range = range(1, code_len+1)

code_1 = lb.mixed_primitive_root_code(pq_set, 1)
# code_2 = lb.mixed_primitive_root_code(pq_set, 1)

def kasami_decimated(m_code: np.ndarray):
	N = m_code.shape[0]
	n = int(math.log2(N+1))
	nd2p1 = 2**(n//2)+1
	code = []
	for i in range(N):
		code.append(m_code[(nd2p1*(i+1)-1)%N])
	return np.array(code)

# m1 = lb.m_code(4, [1])

# code_1 = m1*kasami_decimated(m1)
# code_2 = m1*np.roll(kasami_decimated(m1), 1)

# np.savetxt("a.csv", np.array([
# 	[
# 		np.max(np.abs(lb.cross_correlations(lb.mixed_primitive_root_code(pq_set, k_1), lb.mixed_primitive_root_code(pq_set, k_2)))) for k_1 in k_range
# 	] for k_2 in k_range
# ]), delimiter="\t") # fmt='%.5f'

fig, ax = plt.subplots()

# 相関をプロット
# ax.plot(np.abs(lb.cross_correlations(code_1, code_1)), marker='o', lw=1, color="orange")
# ax.set_title(f"cross correlation")
# ax.set_xlabel("lag (n0)")
# ax.set_ylabel("correlation")
# fig.tight_layout()
# plt.show()

# # IQをプロット
# lb.plt.iq(ax, code_1, s=5, lw=1)
# ax.set_title(f"(p,q)={','.join(map(str, pq_set))}")
# plt.show()

# ## return map
code_target = code_1.imag
ax.scatter(code_target[:-1], code_target[1:], s=4)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("Y[n]")
ax.set_ylabel("Y[n+1]")
for i in range(code_len-2):
	xrange = (0.5, (code_target[i]+1)/2)
	ax.axhline(code_target[i+1], min(xrange), max(xrange), color="black", lw=0.1)
	yrange = (0.5, (code_target[i+1]+1)/2)
	ax.axvline(code_target[i], min(yrange), max(yrange), color="black", lw=0.1)
ax.set_title(f"imaginary part")
fig.tight_layout()
plt.show()