import lb, numba, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000)

def phi(pq: lb.pq, t: int) -> int:
    if t % pq[0] == 0: return 0
    return pow(pq[1], (t % pq[0] - 1), pq[0])

pq = (7, 3)
phis = lb.args_idx(lb.primitive_root_code(pq[0], pq[1], 1), pq[0])

def pmod(x):
    return np.mod(x, pq[0])

k1 = 1
k2 = 3
t0 = pq[0] - 2

# for k1, k2 in [(1, 1), (2, 2), (1, 2)]:
#     print(k1, k2)
#     for t0 in range(3):
#         print(np.mod(phis*k1 - np.roll(phis, -t0)*k2, pq[0]))

for k1 in range(1, pq[0]):
    for k2 in range(1, pq[0]):
        if k1 == (k2*phi(pq, t0+1))%pq[0]: 
            print(pq, k1, k2, np.mod(phis*k1 - np.roll(phis, -t0)*k2, pq[0]))
