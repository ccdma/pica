import numba
import numpy as np
import math
from .misc import *
from scipy.signal import max_len_seq

"""
M系列
n: 原始多項式の最大次数
taps: 原始多項式で係数が1となる次数を列挙(最大次数と定数項は不要)
"""
def m_code(n, taps=None):
	mls_b = max_len_seq(n, state=None, taps=taps)[0] # M系列を生成
	return np.sign(mls_b - 0.5, dtype=np.float64) # {0,1}のビットから{-1,1}のビットに変換

"""
ワイル系列を生成
return ndarray(dtype=complex)
https://www.jstage.jst.go.jp/article/transcom/advpub/0/advpub_2017EBP3139/_article/-char/ja/
"""
@numba.njit("c16[:](f8,f8,i8)")
def weyl_code(low_k: float, delta_k: float, length: int) -> np.ndarray:
	result = []
	for n in range(length):
		x_raw = n*low_k + delta_k
		x = x_raw - math.floor(x_raw)
		result.append(np.exp(2j*np.pi*x))
	return np.complex128(result)

"""
複素数系列を生成
https://www.jstage.jst.go.jp/article/japannctam/55/0/55_0_81/_pdf/-char/ja
n: 何倍角の系列か
rad_0: 初期偏角(ラジアン)
return: exp[rad_0*n^j] j=0,1,2...
"""
@numba.njit("c16[:](i8,f8,i8)")
def const_power_code(n: int, rad_0: float, length: int) -> np.ndarray:
	result = []
	prev = rad_0%(2*np.pi)
	for i in range(length):
		result.append(np.exp(1j*prev))
		prev = (prev * n)%(2*np.pi)
	return np.complex128(result)

"""
チェビシェフ系列を生成（第一種）
deg: チェビシェフ多項式の次数
a0: 初期値
length: 系列の長さ
"""
@numba.njit("f8[:](i8,f8,i8)")
def chebyt_code(n: int, a0: float, length: int) -> np.ndarray:
	return const_power_code(n, np.cos(a0), length).real.astype(np.float64)


"""
原子根符号
int型でmodをとって計算するのでexactに計算可能
const_powerd_samplesだと誤差が出る
"""
@numba.njit("c16[:](i8,i8,i8)")
def primitive_root_code(p: int, q: int, k: int=1) -> np.ndarray:
	result = [1.0]
	prev = k
	for i in range(p-1):
		result.append(np.exp(-1j*2*np.pi*prev/p))
		prev = (prev * q)%p
	return np.complex128(result)

"""
原始根^n符号
pq_list: (p,q)のリスト

2コの場合:
k: N
θn = q^n / p
Ψn = q'^n / p'
Xn = exp(-2j*π*θn*k)*exp(-2j*π*Ψn*k)
"""
def mixed_primitive_root_code(pq_set: list[pq], k: int) -> np.ndarray:
	code_len = math.lcm(*map(lambda pq: pq[0], pq_set))
	code = np.ones(code_len, dtype=np.complex128)
	for p,q in pq_set:
		code *= np.tile(primitive_root_code(p, q, k), code_len//p)
	return code

"""
原始根^n符号（先頭に1を含まない）
"""
def mixed_primitive_root_code_without1(pq_set: list[pq], k: int) -> np.ndarray:
	code_len = math.lcm(*map(lambda pq: pq[1]-1, pq_set))
	code = np.ones(code_len, dtype=np.complex128)
	for p,q in pq_set:
		code *= np.tile(primitive_root_code(p, q, k)[1:], code_len//(p-1))
	return code
