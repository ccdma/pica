import numpy as np
import lb, itertools
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/5426908/find-unique-elements-of-floating-point-array-in-numpy-with-comparison-using-a-d
def funique(arr):
    TOL = 1.0e-7
    brr = arr.copy()
    brr.sort()
    d = np.append(True, np.diff(brr))
    return brr[d>TOL]

def allcomb(arr):
    tmp = []
    for r in range(1, len(arr)+1):
        tmp += list(itertools.combinations(arr, r))
    return tmp

n = 5 # 原始多項式の最大次数
taps_all = list(range(1, n))

for taps1 in allcomb(taps_all):
    m1 = lb.m_code(n, taps=taps1)
    for taps2 in allcomb(taps_all):
        m2 = lb.m_code(n, taps=taps2)
        corr = lb.cross_correlations(m1, m2)
        if len(funique(corr)) == 3:
            print(taps1, taps2)

# plt.plot(lb.cross_correlations(lb.m_code(n, taps=[4,3,2]), lb.m_code(n, taps=[2])))
plt.plot(lb.cross_correlations(lb.m_code(n, taps=[4,3,2]), lb.m_code(n, taps=[2])))
plt.xlabel("lag")
plt.show()