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

pq_list = [(101, 2), (3, 2)]
code_1 = lb.mixed_primitive_root_code(pq_list, 1)
code_len = code_1.shape[0]

code_2 = np.roll(lb.mixed_primitive_root_code(pq_list, 2), 0)

# 相関をプロット
plt.plot(np.abs(get_correlations(code_1, code_2)))
plt.title(f"correlation of X(k1=1) and X(k2)")
plt.xlabel("roll")
plt.ylabel("correlation")
plt.tight_layout()
plt.show()

# # IQをプロット
# plt.scatter(code_1.real, code_1.imag, s=2)
# plt.plot(code_1.real, code_1.imag, lw=2.5)
# plt.scatter(code_2.real, code_2.imag, s=2)
# plt.plot(code_2.real, code_2.imag, lw=1.0)
# plt.title(f"IQ plot of (p, q)={pq_list}: X(k1=1) and X(k2=2,roll=1)")
# plt.gca().set_aspect('equal','datalim')
# plt.show()
