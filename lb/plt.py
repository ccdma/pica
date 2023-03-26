import matplotlib.pyplot as plt
import numpy as np

"""
IQプロットを生成
"""
def iq(ax: plt.Axes, code: np.ndarray, s: float=0.4, lw: float=0.2):
    ax.scatter(code.real, code.imag, s=s)
    ax.plot(code.real, code.imag, lw=lw)
    ax.set_aspect('equal')
