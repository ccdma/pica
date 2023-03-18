import lb, numba, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000)

phi_m = lb.args_index(lb.primitive_root_code(pq[0], pq[1], 1), pq[0])
k1 = 1
k2 = 2
t0 = 1
print(np.mod(phi_m*k1 - np.roll(phi_m, -t0)*k2, pq[0]))