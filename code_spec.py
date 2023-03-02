"""
原始根^n符号の複数個の特性
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

"""
自己相関を計算（冒頭の1を含まない）
"""
def self_correlations(code):
	return lb.cross_correlations(code, code)[1:]

q = 2
pq_set = [(11, 2), (3, 2)]

code_len = lb.mixed_primitive_root_code(pq_set, 1).shape[0]

k_range = range(1, code_len+1)

code_1 = lb.mixed_primitive_root_code(pq_set, 1)
code_2 = lb.mixed_primitive_root_code(pq_set, 2)

# 相関をプロット
# plt.plot(np.abs(lb.cross_correlations(code_1, code_1)), lw=1)
# plt.title(f"cross correlation of X(from=sqrt(2))")
# plt.xlabel("roll")
# plt.ylabel("correlation")
# plt.show()

# # IQをプロット
plt.scatter(code_1.real, code_1.imag, s=1.0)
plt.plot(code_1.real, code_1.imag, lw=0.4)
# plt.title(f"IQ plot of (p, q)={pq_set}: X(k1=1) and X(k2=2,roll=1)")
plt.gca().set_aspect('equal','datalim')
plt.show()
