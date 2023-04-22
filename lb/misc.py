import numpy as np
import random, math
import numba
from typing import TypeAlias, Iterator
from scipy.special import erfc

"""
原始根 (p,q)
"""
pq: TypeAlias = tuple[int, int]

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
SN比を計算 [dB]
https://ja.wikipedia.org/wiki/SN%E6%AF%94
"""
def snr_of(code, noise):
	return 10 * (log_mean_power(code) - log_mean_power(noise))

"""
SNRを指定してガウスノイズ行列を生成

code: 混合後の信号
snr: dB
"""
def gauss_matrix_by_snr(code, snr: float, shape=None):
	noise_log_mean_power = log_mean_power(code) - snr/10
	noise_mean_power = 10**noise_log_mean_power
	stddev = np.sqrt(noise_mean_power/2)
	if not shape:
		shape = code.shape
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
素数判定
"""
def is_prime(n):
    n = abs(n)
    if n == 2: return True
    if n < 2 or n&1 == 0: return False
    return pow(2, n-1, n) == 1

"""
素数判定（is_prime()より遅いのでnumba内での呼び出し時のみ使用）
"""
@numba.njit("b1(i8)")
def x_is_prime(n):
	sqrt_n = int(math.sqrt(n))
	for i in range(2, sqrt_n+1):
		if n%i == 0: return False
	return True

"""
原子根かどうかを判定する
"""
@numba.njit("b1(i8,i8)")
def is_primitive_root(p: int, q: int) -> bool:
	if q >= p:
		return False
	if p <= 2:
		return False
	if not x_is_prime(p):
		return False
	prev = 1
	for i in range(1, p-1):
		prev = (prev*q)%p
		if prev == 1:
			return False
	return True

"""
与えられた範囲で原始根のリスト(p,q)を返す
"""
def find_pq(p_iter: Iterator[int], q_iter: Iterator[int]) -> list[pq]:
	founds: list[pq] = []
	for q in q_iter:
		for p in p_iter:
			if is_primitive_root(p, q):
				founds.append((p, q))
	return founds

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

"""
seedを一括で設定する
"""
def set_seed(x):
	np.random.seed(x)
	random.seed(x)

"""
2つの系列の相互相関を求める
"""
def cross_correlations(code_1: np.ndarray, code_2: np.ndarray):
	code_len = code_1.shape[0]
	code_1_expand = np.tile(code_1[None], (code_len, 1))
	code_2_expand = each_row_roll(np.tile(code_2[None], (code_len, 1)), np.arange(code_len))
	return np.mean(code_1_expand * np.conj(code_2_expand), axis=1) # np.vdot

"""
0~2πにおいてsepで分割した際のindexを評価
"""
def args_idx(const_power_code: np.ndarray, sep: int):
	args = -np.angle(const_power_code)
	idxes = np.mod(np.rint(args/(2*np.pi) * sep), sep) 
	return np.where(idxes == sep, 0, idxes)

"""
与えられたリストをシャッフルしたものを返す
"""
def shuffled(x):
	xcopy = list(x)
	random.shuffle(xcopy)
	return xcopy

"""
SINRよりBERを計算

sinr: array_like
"""
def ber(sinr):
	return 1/2 * erfc(np.sqrt(sinr))
