import lib.ica as ica
import numpy as np
import matplotlib.pyplot as plt
import numba

np.random.seed(0)

def cdma():
    K = 56 # number of users
    N = 61 # code length
    stddev = 0.1
    bits = ica.random_bits([1, K])
    bpsk_data = np.complex64(bits)
    
    B = np.repeat(bpsk_data, N, axis=0).T
    S = np.array([ica.primitive_root_code(N, 2, i+1, True) for i in range(K)])
    
    T = B * S
    A = np.ones(K)
    AWGN = np.random.normal(0, stddev, N) + 1j*np.random.normal(0, stddev, N)

    X = T.T @ A + AWGN

    RB = np.repeat(X[None], K, axis=0)*np.conjugate(S)
    
    rbpsk_data = np.mean(RB, axis=1)
    rbits = np.sign(rbpsk_data.real)

    ber = ica.bit_error_rate(bits, rbits)

    pass

for _ in range(100000):
    cdma()

