"""
符号の特性を観察する
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

"""
(p,q)が原始根となるpを取得する
"""
def find_p_xxx(q: int=2):
	p_list = []
	for p in range(q+1, 100):
		if p == 3: continue
		if lb.is_primitive_root(p, q): p_list(p)
	return p_list

q = 2
pq_list = [(3, q),(5, q),]

code_len = lb.mixed_primitive_root_code(pq_list, 1).shape[0]

c_max = []
for k1 in range(1, code_len):

	each_c_max = []
	c_max.append(each_c_max)
	code_1 = lb.mixed_primitive_root_code(pq_list, k1)
	code_len = code_1.shape[0]

	k2_range = range(1, code_len)
	
	for k2 in k2_range:
		code_2 = lb.mixed_primitive_root_code(pq_list, k2)
		cc = lb.cross_correlations(code_1, code_2)
		each_c_max.append(np.max(np.abs(cc)))

c_max = np.array(c_max)

# print(c_max)
np.savetxt("a.csv", c_max, delimiter=",")

# 相関をプロット
# plt.scatter(k2_range, c_max, s=2)
# plt.plot(k2_range, c_max, lw=1)
# plt.title(f"correlation of X(k1={k1}) and X(k2)")
# plt.xlabel("k2")
# plt.ylabel("correlation")
# plt.tight_layout()
# plt.show()

# # IQをプロット
# plt.scatter(code_1.real, code_1.imag, s=1)
# plt.plot(code_1.real, code_1.imag, lw=1.0)
# plt.title(f"IQ plot of (p, q)={pq_list}: X(k1=1) and X(k2=2,roll=1)")
# plt.gca().set_aspect('equal','datalim')
# plt.show()
