"""
ICAを用いたシュミレーション
"""
import matplotlib.pyplot as plt
import numpy as np
import lb
import dataclasses, sys, warnings, multiprocessing, time, math
from dataclass_csv import DataclassWriter
import concurrent.futures as futu
import random as rand

DELIMITER=","
MAX_WORKERS = multiprocessing.cpu_count()-1

lb.set_seed(0)

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
		ber = math.inf
		snr = math.inf
		if self.complete:
			ber = self.ber_sum/self.complete
			snr = self.snr_sum/self.complete		
		return SummaryReport(
			K=self.K,
			N=self.N,
			ber=ber,
			snr=snr,
			complete=self.complete,
			time=time.perf_counter()-self.start_time
		)

"""
K: number of Users
N: code length
"""
def ica(K: int, N: int, snr: float, _async: bool, seed: int):
	lb.set_seed(seed)

	B = lb.random_bits([K, N])

	# S = np.array([lb.primitive_root_code(N, 2, k) for k in rand.sample(range(1, N+1), K)])
	# S = np.tile(np.array([lb.mixed_primitive_root_code([(3, 2), (5, 2)], k) for k in rand.sample(range(1, K+1), K)]), N//15)
	S = np.array([lb.const_power_code(2, np.random.rand(), N) for k in range(1, K+1)])
	# S = np.array(np.exp(1j*2*np.pi*np.random.rand(K, N)))	# 疑似乱数
	
	ROLL = np.random.randint(0, N, K) if _async else np.zeros(K, dtype=int)	# shape=(K)

	T = B * lb.each_row_roll(S, ROLL)

	A = lb.random_matrix(K)
	MIXED = A @ T

	AWGN = lb.d_gauss_matrix_by_snr(MIXED, snr, MIXED.shape)	# FIXME: mixする前にすべきな気がする / 実部のみにかける
	X = MIXED + AWGN

	real_ica_result = lb.fast_ica_by_sklearn(X.real)
	imag_ica_result = lb.fast_ica_by_sklearn(X.imag)

	real_P = lb.estimate_circulant_matrix(A, real_ica_result.W)
	imag_P = lb.estimate_circulant_matrix(A, imag_ica_result.W)

	Z = real_P.T @ real_ica_result.Y + imag_P.T @ imag_ica_result.Y * 1j

	RB = np.sign(Z.real*lb.each_row_roll(S, ROLL).real + Z.imag*lb.each_row_roll(S, ROLL).imag)
	ber = lb.bit_error_rate(B, RB)

	return EachReport(ber=ber, snr=lb.snr_of(MIXED, AWGN))

def main():
	DataclassWriter(sys.stdout, [], SummaryReport, delimiter=DELIMITER).write()

	N = 1000
	expected_snr = 36.0
	_async = False
	for K in range(2, 60):
		accumlator = ReportAccumulator(K, N)
		with futu.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
			futures = [executor.submit(ica, K, N, expected_snr, _async, int(trial*K*N*expected_snr)) for trial in range(1000)]
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
