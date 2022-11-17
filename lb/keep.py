import numpy as np
from scipy.special import erfc

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
