"""
原始根^n符号の特性を観察する
"""
import lb, itertools, random, dataclasses
import numpy as np
import matplotlib.pyplot as plt

@dataclasses.dataclass
class CodeGen:
	pq_set: list[lb.pq]
	k: int

	def code(self):
		return lb.mixed_primitive_root_code(self.pq_set, self.k)
	
	def __str__(self) -> str:
		return f"(p,q)={self.pq_set[0]},{self.pq_set[0]}"

gens: list[CodeGen] = [CodeGen([(3, 2), (5, 2)], 1), CodeGen([(5, 2), (7, 3)], 1)]
fig, axes = plt.subplots(ncols=len(gens), nrows=1, squeeze=False)	# plt.Figure, plt.Axes[]

for i in range(len(gens)):
	code_1 = gens[i].code()
	ax = axes[0][i]
	lb.plt.iq(ax, code_1, s=3, lw=1)
	ax.set_title(f"{gens[i]}", fontsize=10)

fig.tight_layout()
plt.show()
