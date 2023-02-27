"""
アニメーションを作成

convert -delay 50 -loop 0 *.png movie.gif 
"""
import lb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

COLORS = list(colors.XKCD_COLORS.items())[:100]

fig, ax = plt.subplots()

for i, code_len in enumerate([50, 100, 500, 1000, 5000]):
	code_1 = lb.const_power_code(3, np.sqrt(2), code_len)
	args = np.angle(code_1)
	ax.hist(args, density=True, color=COLORS[i][0])
	ax.set_title(f"length={code_len}")
	ax.set_xlabel("theta(angle)")
	ax.set_ylabel("frequency")
	ax.set_ylim(0, 0.35)

	fig.savefig(f"tmp/{i+1}.png")
	ax.cla()	# reset plot
