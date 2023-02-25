"""
原始根^n符号の特性を観察する
"""
import lb, itertools, random
import numpy as np
import matplotlib.pyplot as plt

"""
原始根を探索
"""
def find_pq(max_p: int, max_q: int) -> list[lb.pq]:
	founds: list[lb.pq] = []	
	for q in range(2, max_q+1):
		for p in range(2, max_p+1):
			if lb.is_primitive_root(p, q):
				founds.append((p, q))
	return founds

pq_comb = itertools.combinations(find_pq(15, 5), 2)
pq_comb = list(filter(lambda pq_set: pq_set[0][0] != pq_set[1][0], pq_comb))
random.shuffle(pq_comb)

fig, axes = plt.subplots(ncols=6, nrows=3)	# plt.Figure, plt.Axes[]

for ax, pq_set in zip(
		itertools.chain.from_iterable(axes),
		pq_comb
	):
	code_1 = lb.mixed_primitive_root_code(pq_set, 1)
	ax.scatter(code_1.real, code_1.imag, s=0.8)
	ax.plot(code_1.real, code_1.imag, lw=0.2)
	ax.set_title(f"{pq_set}")
	ax.set_aspect('equal')

plt.show()
