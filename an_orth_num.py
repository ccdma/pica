import lb, itertools, random
import numpy as np

# p = (N/p1-1)*(N-N/p1+1) + (N/p2-1)*(N-N/p2+1) + (N-1) + (N-N/p1-N/p2+1)*(N/p1+N/p2-1)
def count(p1: int, p2: int):
    return (2*p2-2)*p1**2 + (2*p2**2-4*p2+4)*(p1-1)

for pq1, pq2 in lb.shuffled(itertools.combinations(lb.find_pq(range(4, 20), range(2, 10)), 2)):
    code_num = pq1[0]*pq2[0]

    corr = np.empty((code_num, code_num), dtype=float)
    for k1 in range(code_num):
        for k2 in range(code_num):
            code_1 = lb.mixed_primitive_root_code([pq1, pq2], k1)
            code_2 = lb.mixed_primitive_root_code([pq1, pq2], k2)
            corr[k1, k2] = np.max(np.abs(lb.cross_correlations(code_1, code_2)))
    
    corr_num = np.count_nonzero(corr < 10e-10)

    print(f"(p,q)={pq1},{pq2} analytic={count(pq1[0], pq2[0])}, expect={corr_num}")
