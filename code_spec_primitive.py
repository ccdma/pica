"""
原始根^n符号の特性を観察する
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

"""
自己相関を計算（冒頭の1を含まない）
"""
def self_correlations(code):
	return lb.cross_correlations(code, code)[1:]

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
pq_list = [(11, 2), (3, 2)]

code_len = lb.mixed_primitive_root_code(pq_list, 1).shape[0]

k_range = range(1, code_len+1)

# code_1 = lb.mixed_primitive_root_code(pq_list, 2)
# code_2 = lb.mixed_primitive_root_code(pq_list, 3)

# # 相互相関が最大となる値を集めたk^2テーブル
# c_max = []
# for k1 in k_range:
# 	each_c_max = []
# 	c_max.append(each_c_max)
# 	code_1 = lb.mixed_primitive_root_code(pq_list, k1)
# 	code_len = code_1.shape[0]
# 	k2_range = range(1, code_len)
	
# 	for k2 in k_range:
# 		code_2 = lb.mixed_primitive_root_code(pq_list, k2)
# 		cc = lb.cross_correlations(code_1, code_2)
# 		each_c_max.append(np.max(np.abs(cc)))
# c_max = np.array(c_max)

# # 平均的な自己相関
# self_corr_max = []
# for k1 in k_range:
# 	code = lb.mixed_primitive_root_code(pq_list, k1)
# 	self_corr_max.append(np.mean(np.abs(self_correlations(code))))

# csv書き出し
# np.savetxt("a.csv", c_max, delimiter=",")

code_1 = lb.mixed_primitive_root_code([(173, 2)], 1)[1:]
code_2 = lb.mixed_primitive_root_code([(173, 2)], 2)[1:]

print(np.abs(lb.cross_correlations(code_1, code_1)))
# 相関をプロット
plt.plot(np.abs(lb.cross_correlations(code_1, code_1)), lw=1)
# plt.title(f"cross correlation of X(from=sqrt(2))")
plt.xlabel("roll")
plt.ylabel("correlation")
plt.show()

# # IQをプロット
# plt.scatter(code_1.real, code_1.imag, s=0.8)
# plt.plot(code_1.real, code_1.imag, lw=0.2)
# plt.title(f"IQ plot of (p, q)={pq_list}: X(k1=1) and X(k2=2,roll=1)")
# plt.gca().set_aspect('equal','datalim')
# plt.show()
