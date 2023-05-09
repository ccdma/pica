import lb
import random as rand
import numpy as np
import matplotlib.pyplot as plt

N = 1000000
K = 100
_async = True
snr = 0.01
seed = 1

lb.set_seed(seed)

bits = lb.random_bits([1, K])
bpsk_data = np.complex64(bits)

B = np.repeat(bpsk_data, N, axis=0).T	# shape=(K, N)
# S = np.array([lb.weyl_code(low_k=np.random.rand(), delta_k=np.random.rand(), length=N) for _ in range(1, K+1)])
# S = np.array([lb.mixed_primitive_root_code([(3, 2), (5, 2)], k) for k in rand.sample([1, 2, 3], K)])
S = np.array([lb.const_power_code(2, np.random.rand(), N) for _ in range(1, K+1)])

ROLL = np.random.randint(0, N, K) if _async else np.zeros(K, dtype=int)	# shape=(K)

T = B * lb.each_row_roll(S, ROLL)

A = np.ones(K)
MIXED = T.T @ A
AWGN = lb.gauss_matrix_by_snr(S, snr, MIXED.shape)

print(lb.snr_of(S, AWGN))
print(snr)
