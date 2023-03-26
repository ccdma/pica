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
		return f"(pq:{self.pq_set},k:{self.k})"

gens: list[CodeGen] = []
for pq in lb.find_pq([11], range(2, 7)) :
	for k in range(1, 4):
		gens.append(CodeGen([(3, 2), pq,], k))

fig, axes = plt.subplots(ncols=len(gens), nrows=len(gens), squeeze=False)	# plt.Figure, plt.Axes[]

for i in range(len(gens)):
	for j in range(len(gens)):
		code_1 = gens[i].code()
		code_2 = gens[j].code()
		ax = axes[i][j]
		ax.plot(np.abs(lb.cross_correlations(code_1, code_2)))
		ax.set_ylim(0, 1)
		ax.set_title(f"{gens[i]}x{gens[j]}", fontsize=8)

fig.tight_layout(h_pad=-0.9, w_pad=-0.5)
plt.show()
