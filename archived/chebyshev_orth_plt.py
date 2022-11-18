"""
チェビシェフ多項式の直行性をグラフで確認
"""
import lb
import matplotlib.pyplot as plt
import numpy as np

initials = np.linspace(-0.01, 0.01, 100)
xx, yy = np.meshgrid(initials, initials)
cps = np.array([lb.const_power_code(2, i, 1000).real for i in initials])
c = lb.correlation(cps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, c, cmap='bwr', linewidth=0)
fig.colorbar(surf)
ax.set_title("Surface Plot")
fig.show()
plt.show()
pass