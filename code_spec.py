"""
符号の特性を観察する
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

# def find_group(G):
# 	def dfs(v, G, seen, group):
# 		group.append(v)
# 		# 頂点 v を探索済みにする
# 		seen[v] = True
# 		# G[v] に含まれる頂点 v2 について、
# 		for v2, val in enumerate(G[v]):
# 			# 直交していないのでスキップ
# 			if val > 1e-10: continue
# 			# v2 がすでに探索済みならば、スキップする
# 			if seen[v2]: continue
# 			# v2 始点で深さ優先探索を行う (関数を再帰させる)
# 			dfs(v2, G, seen, group)
# 		return
# 	N = G.shape[0]
# 	seen = [False for _ in range(N)]    # seen[v]：頂点 v が探索済みなら true, そうでないなら false
# 	groups = []
# 	# 全ての頂点について
# 	for v in range(N):
# 		# 頂点 v がすでに訪問済みであれば、スキップ
# 		if seen[v]: continue
# 		# そうでなければ、頂点 v を含む連結成分は未探索
# 		# 深さ優先探索で新たに訪問し、答えを 1 増やす
# 		group = []
# 		dfs(v, G, seen, group)
# 		groups.append(group)
# 	return groups

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
p_b = 11
pq_list = [(p_b, 2), (3, 2)]

code_1 = lb.mixed_primitive_root_code(pq_list, 1)
code_len = code_1.shape[0]
c = []
for k in range(1, code_len):
	code_2 = lb.mixed_primitive_root_code(pq_list, k)
	c.append(np.max(np.abs(lb.cross_correlations(code_1, code_2))))

# 相関をプロット
# plt.plot(c)
# plt.title(f"correlation of X(k1=1) and X(k2)")
# plt.xlabel("roll")
# plt.ylabel("correlation")
# plt.tight_layout()
# plt.show()

# # IQをプロット
plt.scatter(code_1.real, code_1.imag, s=1)
plt.plot(code_1.real, code_1.imag, lw=1.0)
plt.title(f"IQ plot of (p, q)={pq_list}: X(k1=1) and X(k2=2,roll=1)")
plt.gca().set_aspect('equal','datalim')
plt.show()
