"""
ICAを用いたシュミレーション
"""
import matplotlib.pyplot as plt
import numpy as np
import lb
import dataclasses, sys, warnings, multiprocessing, time
from dataclass_csv import DataclassWriter
import concurrent.futures as futu

DELIMITER="\t"
MAX_WORKERS = multiprocessing.cpu_count()-1

np.random.seed(0)

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
	time: float

@dataclasses.dataclass
class ReportAccumulator:
	K: int
	N: int
	ber_sum = 0
	snr_sum = 0
	complete = 0
	start_time = time.perf_counter()

	def add(self, report: EachReport):
		self.ber_sum += report.ber
		self.snr_sum += report.snr
		self.complete += 1
	
	def summary(self):
		return SummaryReport(
			K=self.K,
			N=self.N,
			ber=self.ber_sum/self.complete,
			snr=self.snr_sum/self.complete,
			complete=self.complete,
			time=time.perf_counter()-self.start_time
		)

"""
K: number of Users
N: code length
"""
def ica(K: int, N: int, snr: float, seed: int):
	np.random.seed(seed)

	B = lb.random_bits([K, N])

	# S = np.array([lb.mixed_primitive_root_code([(53, 2), (19, 2)], k) for k in range(1, K+1)])
	# S = np.array([lb.primitive_root_code(N, 2, k) for k in range(1, K+1)])
	S = np.array([lb.const_power_code(2, np.random.rand(), N) for k in range(1, K+1)])
	
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

def main():
	DataclassWriter(sys.stdout, [], SummaryReport, delimiter=DELIMITER).write()

	N = 1007
	expected_snr = 30
	for K in range(2, N):
		accumlator = ReportAccumulator(K, N)
		with futu.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
			futures = [executor.submit(ica, K, N, expected_snr, trial) for trial in range(1000)]
			for future in futu.as_completed(futures):
				try:
					report = future.result()
					accumlator.add(report)
				except Warning as e:
					pass
		DataclassWriter(sys.stdout, [accumlator.summary()], SummaryReport, delimiter=DELIMITER).write(skip_header=True)

if __name__ == '__main__':
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		main()