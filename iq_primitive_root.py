"""
原始根符号によるパワー一定符号のIQプロット
"""
import lb
import matplotlib.pyplot as plt
import numpy as np

p = 1024
q = 2
k = 1

nrows = 1
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5,nrows*5))
# axes = list(itertools.chain.from_iterable(axes))
fig.suptitle(f"primitive root code (generated)", fontsize=12)
fig.gca().set_aspect('equal','datalim')

def plot_primitive(p, q, ax):
	samplings = lb.primitive_root_code(p, q)[1:]
	ax.scatter(samplings.real, samplings.imag, s=1)
	ax.plot(samplings.real, samplings.imag, lw=0.2)
	# ax.scatter(samplings.real[0], samplings.imag[0])
	# ax.scatter(samplings.real[-1], samplings.imag[-1])
	ax.set_title(f"p={p},q={q}")

plot_primitive(173, 59, axes[0])
plot_primitive(173, 45, axes[1])
plot_primitive(173, 46, axes[2])
# plot_primitive(173, 2, axes[3])
# plot_primitive(389, 2, axes[4])
# plot_primitive(1019, 2, axes[5])

fig.tight_layout()
plt.show()