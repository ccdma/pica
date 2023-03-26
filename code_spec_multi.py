"""
原始根^n符号の特性を観察する
"""
import lb, itertools, random
import numpy as np
import matplotlib.pyplot as plt

pq_comb = list(itertools.combinations(lb.find_pq(range(2, 10), range(2, 4)), 2))
# pq_comb = list(filter(lambda pq_set: pq_set[0][0] != pq_set[1][0], pq_comb))
random.shuffle(pq_comb)

fig, axes = plt.subplots(ncols=1, nrows=1, squeeze=False)	# plt.Figure, plt.Axes[]

for ax, pq_set in zip(
		itertools.chain.from_iterable(axes),
		pq_comb
	):
	code_1 = lb.mixed_primitive_root_code(pq_set, 1)
	lb.plt.iq(ax, code_1)
	ax.set_title(f"{pq_set}")

plt.show()
