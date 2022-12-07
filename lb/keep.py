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
stddev=0の場合、シュミレーションとの一致を確認

N: code of length (array_like)
stddev: noise stddev
K: number of users (array_like)
"""
def cdma_ber(N, stddev, K: np.array):
    return 1/2 * erfc(N/np.sqrt((K-1)*N + stddev**2))
