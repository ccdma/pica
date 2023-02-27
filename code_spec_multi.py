"""
原始根^n符号の特性を観察する
"""
import lb, itertools, random
import numpy as np
import matplotlib.pyplot as plt

pq_comb = list(itertools.combinations(lb.find_pq(range(20), [2]), 2)) + list(itertools.combinations(lb.find_pq(range(20), [3]), 2))
pq_comb = list(filter(lambda pq_set: pq_set[0][0] != pq_set[1][0], pq_comb))
# random.shuffle(pq_comb)

fig, axes = plt.subplots(ncols=5, nrows=2)	# plt.Figure, plt.Axes[]

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
