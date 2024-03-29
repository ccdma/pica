import numpy as np
import math

P = 13
DATA = np.arange(P)
EXP = np.exp(2j * np.pi * np.arange(P) / P)

def comb(r, start, combset):
    if r == 0:
        _sum = np.abs(np.sum(EXP[combset]))
        if _sum < 10e-10:
            print(combset, _sum)
        return

    for i in range(start, P):
        comb(r - 1, i, combset + [DATA[i]])

print(math.comb(P*2-1, P))
comb(P, 0, [])
