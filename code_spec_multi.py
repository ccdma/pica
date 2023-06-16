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
		# return f"(p,q)={','.join(map(str, self.pq_set))}"
		return f"k={self.k}"

plt.rcParams["figure.figsize"] = [18,12]

gens: list[CodeGen] = [CodeGen([(7,3)], k) for k in range(1, 8)]
fig, axes = plt.subplots(ncols=len(gens), nrows=len(gens), squeeze=False)	# plt.Figure, plt.Axes[]

for i in range(len(gens)):
	for j in range(len(gens)):
		code_1 = gens[i].code()
		code_2 = gens[j].code()
		ax = axes[i][j]
		ax.plot(np.abs(lb.cross_correlations(code_1, code_2)), marker='o', lw=1)
		ax.set_title(f"{gens[i]} x {gens[j]}", fontsize=11)
		ax.set_xlabel("lag (n0)")
		ax.set_ylabel("correlation")
		ax.set_ylim(0, 0.5)

fig.suptitle(f"correlation: (p,q)={','.join(map(str, gens[0].pq_set))}", fontsize=14)
fig.tight_layout()
fig.savefig("1.png", dpi=120)