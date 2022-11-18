"""
符号の特性を観察する
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

pq_list = [(61, 2)]
code_1 = lb.mixed_primitive_root_code(pq_list, 1)
code_len = code_1.shape[0]
code_2 = lb.mixed_primitive_root_code(pq_list, 4)

correlations = []
for roll in range(0, code_len):
	correlations.append(np.abs(np.vdot(code_1, np.roll(code_2, roll))) / code_len)

# 相関をプロット
plt.plot(correlations)
plt.title(f"correlation of X(k1=1) and X(k2=2)")
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
