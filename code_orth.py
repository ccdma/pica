import lb, numba, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000)

pq1 = (11, 2)
phi_1 = lb.args_index(lb.primitive_root_code(pq1[0], pq1[1], 1), pq1[0])

pq2 = (3, 2)
phi_2 = lb.args_index(lb.primitive_root_code(pq2[0], pq2[1], 1), pq2[0])

k1 = 2
k2 = 1

# for t0 in range(pq1[0]*pq2[0]):
t0 = 0

i1 = np.mod(phi_1*k1 - np.roll(phi_1, -t0)*k2, pq1[0])
i2 = np.mod(phi_2*k1 - np.roll(phi_2, -t0)*k2, pq2[0])

# print(i1)
print(i2)

# print(np.sum(np.exp(-2j*np.pi*i1/pq1[0]))*np.sum(np.exp(-2j*np.pi*i2/pq2[0])))

for pq1 in lb.find_pq(range(2, 60), range(2, 60)):
    p = pq1[0]
    phi_1 = lb.args_index(lb.primitive_root_code(pq1[0], pq1[1], 1), pq1[0])
    phi = list(map(int, phi_1.tolist()))
    for t1 in range(p):
        for t2 in range(p):
            exp = phi[(t1+t2)%p]
            # print(t1, t2, exp, end=" ")
            if t1 == 0:
                assert exp == (phi[t2])%p
            else:
                if t1 + t2 < p:
                    assert exp == (phi[t1]*phi[(t2+1)%p])%p
                if t1 + t2 == p:
                    assert exp == 0
                if t1 + t2 > p:
                    assert exp == (phi[t1]*phi[t2])%p