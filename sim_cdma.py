"""
CDMAのシュミレーション
"""
import lb
import numpy as np
import matplotlib.pyplot as plt
import dataclasses, sys, warnings, multiprocessing, time
import dataclass_csv
import concurrent.futures as futu

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
K: number of users
N: code length
"""
def cdma(K: int, N: int, snr: float, seed: int) -> EachReport:
	np.random.seed(seed)

	bits = lb.random_bits([1, K])
	bpsk_data = np.complex64(bits)
	
	B = np.repeat(bpsk_data, N, axis=0).T
	# S = np.array([lb.mixed_primitive_root_code([(5, 2), (13, 2)], k) for k in range(1, K+1)])
	# S = np.array([lb.primitive_root_code(N, 2, k) for k in range(1, K+1)])
	S = np.array([lb.const_power_code(2, np.random.rand(), N) for k in range(1, K+1)])

	T = B * S
	A = np.ones(K)
	MIXED = T.T @ A
	AWGN = lb.gauss_matrix_by_snr(MIXED, snr)
	X = MIXED + AWGN

	RB = np.repeat(X[None], K, axis=0)*np.conjugate(S)

	rbpsk_data = np.mean(RB, axis=1)
	rbits = np.sign(rbpsk_data.real)

	ber = lb.bit_error_rate(bits, rbits)

	return EachReport(ber=ber, snr=lb.snr(MIXED, AWGN))

def main():
	delimiter="\t"
	dataclass_csv.DataclassWriter(sys.stdout, [], SummaryReport, delimiter=delimiter).write()

	N = 65
	expected_snr = 5
	for K in range(2, N):
		accumlator = ReportAccumulator(K, N)
		with futu.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
			futures = [executor.submit(cdma, K, N, expected_snr, trial) for trial in range(10000)]
			for future in futu.as_completed(futures):
				try:
					report = future.result()
					accumlator.add(report)
				except Warning as e:
					pass
		dataclass_csv.DataclassWriter(sys.stdout, [accumlator.summary()], SummaryReport, delimiter=delimiter).write(skip_header=True)

if __name__ == '__main__':
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		main()