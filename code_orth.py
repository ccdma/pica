import lb, numba, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000)

def phi(pq: lb.pq, t: int) -> int:
    if t % pq[0] == 0: return 0
    return pow(pq[1], (t % pq[0] - 1), pq[0])


pq = (11, 2)
phis = lb.args_idx(lb.primitive_root_code(pq[0], pq[1], 1), pq[0])

k1 = 2
k2 = 1
t0 = 0

for t0 in range(pq[0]):
    print(f"### {t0}")
    print(f"phi(t0)\t= {phi(pq, t0)}")
    print(f"phi(t0+1)\t= {phi(pq, t0+1)}")
# print(f"-k2 phi(t0)\t= {-k2*phi(pq, t0)}")
