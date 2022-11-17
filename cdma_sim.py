import lib.ica as ica
import numpy as np
import matplotlib.pyplot as plt
import numba
import dataclasses, sys
import dataclass_csv

np.random.seed(0)

@dataclasses.dataclass
class EachReport:
    ber: int

@dataclasses.dataclass
class SummaryReport:
    K: int
    N: int
    stddev: float
    ber: int
    complete: int

@numba.njit("c16[:,:](i8,i8)")
def make_code(N: int, K: int):
    codes = np.empty((K, N), dtype=np.complex128)
    for i in range(K):
        # codes[i] = ica.primitive_root_code(N, 2, i+1, True)
        codes[i] = ica.const_power_code(2, np.random.rand(), N)
    return codes

"""
K: number of users
N: code length
"""
def cdma(K: int, N: int, stddev: float) -> EachReport:
    bits = ica.random_bits([1, K])
    bpsk_data = np.complex64(bits)
    
    B = np.repeat(bpsk_data, N, axis=0).T
    S = make_code(N, K)
    
    T = B * S
    A = np.ones(K)
    AWGN = np.random.normal(0, stddev, N) + 1j*np.random.normal(0, stddev, N)

    X = T.T @ A + AWGN

    RB = np.repeat(X[None], K, axis=0)*np.conjugate(S)
    
    rbpsk_data = np.mean(RB, axis=1)
    rbits = np.sign(rbpsk_data.real)

    ber = ica.bit_error_rate(bits, rbits)

    return EachReport(ber=ber)

N = 61
stddev = 0.1
dataclass_csv.DataclassWriter(sys.stdout, [], SummaryReport).write()
for K in range(2, 61):
    ber_sum = 0
    complete = 0
    for trial in range(10000):
        report = cdma(K, N, stddev)
        ber_sum += report.ber
        complete += 1
    dataclass_csv.DataclassWriter(sys.stdout, [SummaryReport(
        K=K,
        N=N,
        stddev=stddev,
        ber=ber_sum/complete,
        complete=complete
    )], SummaryReport).write(skip_header=True)
