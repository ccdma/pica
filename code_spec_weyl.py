"""
Weyl符号の特性を観察する
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

code_len = 100
code_1 = lb.weyl_code(np.sqrt(0.2), np.sqrt(0.3), code_len)
code_2 = lb.weyl_code(np.sqrt(0.1), np.sqrt(0.5), code_len)

code_x = code_1*code_2
code_y = code_1*np.roll(code_2, 1)

plt.plot(np.abs(lb.cross_correlations(code_1, code_1)), lw=1)
plt.title(f"mean of self correlation of X(k1)")
plt.xlabel("k1")
plt.ylabel("correlation")
plt.tight_layout()
plt.show()

# # IQをプロット
# plt.scatter(code_1.real, code_1.imag, s=1)
# plt.plot(code_1.real, code_1.imag, lw=1.0)
# plt.title(f"IQ plot of weyl")
# plt.gca().set_aspect('equal','datalim')
# plt.show()

