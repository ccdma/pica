import numpy as np

"""
平均二乗誤差
"""
def mse(A: np.ndarray, B: np.ndarray) -> float:
	return ((A - B)**2).mean()

"""
複素信号に対して平均電力のlog10を算出
"""
def log_mean_power(code):
	mean_power = np.mean(np.power(code.real, 2) + np.power(code.imag, 2))
	return np.log10(mean_power)

"""
SN比を計算
"""
def snr(code, noise):
	return 10 * (log_mean_power(code) - log_mean_power(noise))

"""
SNRを指定してガウスノイズ行列を生成

code: 混合後の信号
"""
def gauss_matrix_by_snr(code, snr: float):
	noise_log_mean_power = log_mean_power(code) - snr/10
	noise_mean_power = 10**noise_log_mean_power
	stddev = np.sqrt(noise_mean_power/2)
	return np.random.normal(0, stddev, code.shape) + 1j*np.random.normal(0, stddev, code.shape)

""" 
-0.5~+0.5なる混合行列を作成
size: 正方行列のサイズ
"""
def random_matrix(size: int) -> np.ndarray:
	return np.random.rand(size, size)-0.5

"""
±1のランダム行列を生成
"""
def random_bits(shape) -> np.ndarray:
	return np.sign(np.random.rand(*shape) - 0.5)

"""
±1のビットでBERを計測
"""
def bit_error_rate(bits1, bits2) -> np.ndarray:
	return np.mean(np.abs(bits1 - bits2))/2.0

"""
素数判定
"""
def is_prime(n):
    n = abs(n)
    if n == 2: return True
    if n < 2 or n&1 == 0: return False
    return pow(2, n-1, n) == 1

"""
原子根かどうかを判定する
"""
def is_primitive_root(p: int, q: int) -> bool:
	if q >= p:
		return False
	if p <= 1:
		return False
	if p == 2:
		return True
	if not is_prime(p):
		return False
	prev = 1
	for i in range(1, p-1):
		prev = (prev*q)%p
		if prev == 1:
			return False
	return True

"""
各行に対してrollを行う（非同期CDMAで利用する想定）

>>> A
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> r
array([0, 2])
>>> each_row_roll(A, r)
array([[0, 1, 2, 3, 4],
       [8, 9, 5, 6, 7]])
"""
def each_row_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]