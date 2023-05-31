"""
CDMAのシュミレーション
"""
import lb
import numba
import numpy as np
import matplotlib.pyplot as plt
import dataclasses, sys, warnings, multiprocessing, time
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
	noise: float	# power of noise

@dataclasses.dataclass
class SummaryReport:
	K: int
	N: int
	ber: float
	snr: float
	noise: float
	complete: int
	time: float

@dataclasses.dataclass
class ReportAccumulator:
	K: int
	N: int
	ber_sum = 0
	snr_sum = 0
	noise_sum = 0
	complete = 0
	start_time = time.perf_counter()

	def add(self, report: EachReport):
		self.ber_sum += report.ber
		self.snr_sum += report.snr
		self.noise_sum += report.noise
		self.complete += 1
	
	def summary(self):
		return SummaryReport(
			K=self.K,
			N=self.N,
			ber=self.ber_sum/self.complete,
			snr=self.snr_sum/self.complete,
			noise=self.noise_sum/self.complete,
			complete=self.complete,
			time=time.perf_counter()-self.start_time
		)

"""
自己相関からrollを推定する
"""
@numba.njit("i8[:](c16[:],c16[:,:],i8,i8)")
def estimate_roll(X: np.ndarray, S: np.ndarray, K: int, N: int):
	roll_bpsk = np.empty((K, N), dtype=np.complex128)
	for roll in range(N):
		roll_RB = np.repeat(X, K).reshape((-1, K)).T*np.conjugate(np.roll(S, roll))
		roll_bpsk[:, roll] = np.sum(roll_RB, axis=1)
	return np.argmax(np.abs(roll_bpsk), axis=1).astype(np.int64)

"""
K: number of users
N: code length
async: True=チップ同期 / False=ビット同期
"""
def cdma(K: int, N: int, snr: float, _async: bool, seed: int) -> EachReport:
	lb.set_seed(seed)

	bits = lb.random_bits([1, K])
	bpsk_data = np.complex64(bits)
	
	B = np.repeat(bpsk_data, N, axis=0).T	# shape=(K, N)
	# S = np.array([lb.weyl_code(low_k=np.random.rand(), delta_k=np.random.rand(), length=N) for _ in range(1, K+1)])
	S = np.array([lb.mixed_primitive_root_code([(3, 2), (5, 2)], k) for k in rand.sample([1, 2, 3], K)])
	# S = np.array([lb.const_power_code(2, np.random.rand(), N) for _ in range(1, K+1)])

	ROLL = np.random.randint(0, N, K) if _async else np.zeros(K, dtype=int)	# shape=(K)

	T = B * lb.each_row_roll(S, ROLL)

	A = np.ones(K)
	MIXED = T.T @ A
	AWGN = np.random.normal(0, lb.stddev_of_noise_by_snr(S, snr), MIXED.shape)
	X = MIXED + AWGN

	R_ROLL = ROLL #estimate_roll(X, S, K, N)

	RB = np.repeat(X[None], K, axis=0)*np.conjugate(lb.each_row_roll(S, R_ROLL))

	rbpsk_data = np.mean(RB, axis=1)
	rbits = np.sign(rbpsk_data.real)

	ber = lb.bit_error_rate(bits, rbits)
	return EachReport(ber=ber, snr=lb.snr_of(S, AWGN), noise=np.power(10, lb.log_mean_power(AWGN)))

_async = True

def do_trial(K, N, snr):
	accumlator = ReportAccumulator(K, N)
	for trial in range(100000000):
		try:
			report = cdma(K, N, snr, _async, trial)
			accumlator.add(report)
		except Warning as e:
			pass
	return accumlator.summary()

N = 15
K = 3
# snr = 30

def main():
	DataclassWriter(sys.stdout, [], SummaryReport, delimiter=DELIMITER).write()

	with futu.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
		futures = [executor.submit(do_trial, K, N, snr) for snr in range(-12, 12, 2)[5:]]
		for future in futu.as_completed(futures):
			DataclassWriter(sys.stdout, [future.result()], SummaryReport, delimiter=DELIMITER).write(skip_header=True)

if __name__ == '__main__':
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		main()