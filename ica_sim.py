"""
ICAを用いてパワー一定カオス拡散符号の復元を行う
"""
import matplotlib.pyplot as plt
import numpy as np
import lb
import dataclasses, sys, warnings
import dataclass_csv

np.random.seed(1)

@dataclasses.dataclass
class EachReport:
	ber: int
	snr: float

@dataclasses.dataclass
class SummaryReport:
	K: int
	N: int
	ber: float
	snr: float
	complete: int

"""
K: number of Users
N: code length
"""
def ica(K: int, N: int, snr: float):
	B = lb.random_bits([K, N])

	S = np.array([lb.primitive_root_code(N, 2, k) for k in range(1, K+1)])
	T = B * S

	A = lb.random_matrix(K)
	MIXED = A @ T

	AWGN = lb.gauss_matrix_by_snr(MIXED, snr)
	X = MIXED + AWGN

	real_ica_result = lb.fast_ica_by_sklearn(X.real)
	imag_ica_result = lb.fast_ica_by_sklearn(X.imag)

	real_P = lb.estimate_circulant_matrix(A, real_ica_result.W)
	imag_P = lb.estimate_circulant_matrix(A, imag_ica_result.W)

	Z = real_P.T @ real_ica_result.Y + imag_P.T @ imag_ica_result.Y * 1j

	RB = np.sign(Z.real*S.real + Z.imag*S.imag)
	ber = lb.bit_error_rate(B, RB)

	return EachReport(ber=ber, snr=lb.snr(MIXED, AWGN))

N = 1019
expected_snr = 10
dataclass_csv.DataclassWriter(sys.stdout, [], SummaryReport).write()
for K in range(2, N):
	ber_sum = 0
	snr_sum = 0
	complete = 0
	for trial in range(1000):
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			try:
				report = ica(K, N, expected_snr)
				ber_sum += report.ber
				snr_sum += report.snr
				complete += 1
			except Warning as e:
				pass
	dataclass_csv.DataclassWriter(sys.stdout, [SummaryReport(
		K=K,
		N=N,
		ber=ber_sum/complete,
		snr=snr_sum/complete,
		complete=complete
	)], SummaryReport).write(skip_header=True)
