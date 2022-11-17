from scipy.special import eval_chebyt, eval_chebyu
import numpy as np

def _test_const_powerd_samples(n: int, rad_0: float, length: int) -> np.ndarray:
	result = []
	a_0 = np.exp(rad_0*1j)
	result.append(a_0)
	for _ in range(length-1):
		a_0 = complex(eval_chebyt(n, a_0.real), eval_chebyu(n-1, a_0.real)*a_0.imag)
		result.append(a_0)
	return np.complex128(result)

"""
チェビシェフ系列を生成（第一種）
deg: チェビシェフ多項式の次数
a0: 初期値
length: 系列の長さ
"""
def _test_chebyt_samples(n: int, a0: float, length: int) -> np.ndarray:
	result = [a0]
	for _ in range(length-1):
		a0 = eval_chebyt(n, a0)
		result.append(a0)
	return np.array(result)

