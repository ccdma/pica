"""
符号の特性を観察する
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
pq_list = [(5, q),(11, q),]

code_len = lb.mixed_primitive_root_code(pq_list, 1).shape[0]
k_range = range(1, code_len+1)

# 相互相関が最大となる値を集めたk^2テーブル
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

# 平均的な自己相関
self_corr_avg = []
for k1 in k_range:
	code = lb.mixed_primitive_root_code(pq_list, k1)
	self_corr_avg.append(np.mean(np.abs(self_correlations(code))))

# csv書き出し
# np.savetxt("a.csv", self_corr_avg, delimiter=",")

# 相関をプロット
plt.scatter(k_range, self_corr_avg, s=2)
plt.plot(k_range, self_corr_avg, lw=1)
# plt.plot(np.abs(self_correlations(lb.mixed_primitive_root_code(pq_list, 5))))
plt.title(f"mean of self correlation of X(k1)")
plt.xlabel("k1")
plt.ylabel("correlation")
plt.tight_layout()
plt.show()

# # IQをプロット
# plt.scatter(code_1.real, code_1.imag, s=1)
# plt.plot(code_1.real, code_1.imag, lw=1.0)
# plt.title(f"IQ plot of (p, q)={pq_list}: X(k1=1) and X(k2=2,roll=1)")
# plt.gca().set_aspect('equal','datalim')
# plt.show()
