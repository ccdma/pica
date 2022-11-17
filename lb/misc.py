import numpy as np
from scipy.special import erfc

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
shape: ノイズ行列の形
"""
def gauss_matrix_by_snr(code, snr: float, shape):
	noise_log_mean_power = log_mean_power(code) - snr/10
	noise_mean_power = 10**noise_log_mean_power
	stddev = np.sqrt(noise_mean_power/2)
	return np.random.normal(0, stddev, shape) + 1j*np.random.normal(0, stddev, shape)

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
2つの行ごとの内積を計算する
"""
def correlation(P: np.ndarray) -> np.ndarray:
	res = np.eye(P.shape[0], dtype=P.dtype)
	for i in range(P.shape[0]):
		for j in range(P.shape[0]):
			res[i][j] = (P[i]@P[j]) / P.shape[1]
	return res

"""
N: code of length (array_like)
sigma: noise size
K: number of users (array_like)
"""
def cdma_ber(N, sigma, K: np.array):
    return 1/2 * erfc(N/np.sqrt(2*((K-1)*N + sigma**2)))
