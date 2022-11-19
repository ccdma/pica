"""
符号の特性を観察する
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

"""
2つの系列の相互相関を求める
"""
def get_correlations(code_1: np.ndarray, code_2: np.ndarray):
	code_len = code_1.shape[0]
	code_1_expand = np.tile(code_1[None], (code_len, 1))
	code_2_expand = lb.each_row_roll(np.tile(code_2[None], (code_len, 1)), np.arange(code_len))
	return np.mean(code_1_expand * np.conj(code_2_expand), axis=1) # np.vdot

pq_list = [(61, 2)]
code_1 = lb.mixed_primitive_root_code_without1(pq_list, 1)
code_len = code_1.shape[0]

correlations = []
for_range = range(2, code_len)
for k in for_range:
	code_2 = lb.mixed_primitive_root_code_without1(pq_list, k)
	correlations.append(np.argmax(np.abs(get_correlations(code_1, code_2))))

# 相関をプロット
plt.plot(for_range, correlations)
# plt.plot(np.abs(get_correlations(code_1, lb.mixed_primitive_root_code_without1(pq_list, 10))))
plt.title(f"correlation of X(k1=1) and X(k2)")
plt.xlabel("roll")
plt.ylabel("correlation")
plt.tight_layout()
plt.show()

# # IQをプロット
# plt.scatter(code_1.real, code_1.imag, s=1)
# plt.plot(code_1.real, code_1.imag, lw=0.2)
# plt.title(f"IQ plot of (p, q)={pq_list}")
# plt.gca().set_aspect('equal','datalim')
# plt.show()
