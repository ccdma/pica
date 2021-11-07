import numpy.linalg as la
import numpy as np
from scipy.special import eval_chebyt, eval_chebyu
import dataclasses

@dataclasses.dataclass
class FastICAResult:
	# Y represents obtained independent data.
	# 
	# EX:
	#   [[y_0(0), y_0(1), y_0(2)]
	#    [y_1(0), y_1(1), y_1(2)]]
	#   s.t. y_point(time)
	Y: np.ndarray
	X_whiten: np.ndarray
	X_center: np.ndarray

"""
X: represents observed data
[[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する
EX) 
   [[x_0(0), x_0(1), x_0(2)]
	[x_1(0), x_1(1), x_1(2)]]
	s.t. x_point(time) 
"""
def FastICA(X: np.ndarray, _assert: bool=True) -> FastICAResult:
	SAMPLE, SERIES = X.shape # (観測点数, 観測時間数)

	# 中心化を行う（観測点ごとの平均であることに注意）
	mean = np.mean(X,axis=1)
	X_center = X - np.array([ np.full(SERIES, ave) for ave in mean ]) 

	# 固有値分解により、白色化されたX_whitenを計算する
	lambdas, P = la.eig(np.cov(X_center))
	if _assert:
		assert np.allclose(np.cov(X_center), P @ np.diag(lambdas) @ P.T) # 固有値分解の検証
	for i in reversed(np.where(lambdas < 1.e-12)[0]): # 極めて小さい固有値は削除する
		lambdas = np.delete(lambdas, i, 0)
		P = np.delete(P, i, 1)
	Atilda = la.inv(np.sqrt(np.diag(lambdas))) @ P.T # 球面化行列
	X_whiten = Atilda @ X_center
	if _assert:
		assert np.allclose(np.cov(X_whiten), np.eye(X_whiten.shape[0]), atol=1.e-10) # 無相関化を確認（単位行列）

	# ICAに使用する関数gとその微分g2（ここではgは４次キュムラント）
	g = lambda bx : bx**3
	g2 = lambda bx : 3*(bx**2)

	I = X_whiten.shape[0]
	B = np.array([[np.random.rand()-0.5 for i in range(I)] for j in range(I) ]) # X_whitenからYへの復元行列

	# Bを直交行列かつ列ベクトルが大きさ１となるように規格化
	for i in range(I):
		if i > 0:
			B[:,i] = B[:,i] - B[:,:i] @ B[:,:i].T @ B[:,i] # 直交空間に射影
		B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化

	# Bの決定(Y = B.T @ X_whiten)
	for i in range(I):
		for j in range(1000):
			prevBi = B[:,i].copy()
			B[:,i] = np.average([*map( # 不動点法による更新
				lambda x: g(x @ B[:,i])*x - g2(x @ B[:,i])*B[:,i],
				X_whiten.T
			)], axis=0)
			B[:,i] = B[:,i] - B[:,:i] @ B[:,:i].T @ B[:,i] # 直交空間に射影
			B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化
			if 1.0 - 1.e-8 < abs(prevBi @ B[:,i]) < 1.0 + 1.e-8: # （内積1 <=> ほとんど変更がなければ）
				break
		else:
			assert False
	if _assert:
		assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1.e-10) # Bが直交行列となっていることを検証

	Y = B.T @ X_whiten

	return FastICAResult(Y=Y, X_whiten=X_whiten, X_center=X_center)


"""
エルミート転置
"""
def _H(P: np.ndarray):
	return np.conjugate(P.T)

@dataclasses.dataclass
class CFastICAResult:
	# Y represents obtained independent data.
	# 
	# EX:
	#   [[y_0(0), y_0(1), y_0(2)]
	#    [y_1(0), y_1(1), y_1(2)]]
	#   s.t. y_point(time)
	Y: np.ndarray

"""
複素数版FastICA
https://www.cs.helsinki.fi/u/ahyvarin/papers/IJNS00.pdf
"""
def CAFastICA(X: np.ndarray, _assert: bool=True) -> CFastICAResult:
	SAMPLE, SERIES = X.shape # (観測点数, 観測時間数)

	# 中心化を行う（観測点ごとの平均であることに注意）
	mean = np.mean(X,axis=1)
	X_center = X - np.array([ np.full(SERIES, ave) for ave in mean ]) 

	# 固有値分解により、白色化されたX_whitenを計算する
	lambdas, P = la.eig(np.cov(X_center))
	if _assert:
		assert np.allclose(np.cov(X_center), P @ np.diag(lambdas) @ _H(P)) # 固有値分解の検証
	for i in reversed(np.where(lambdas < 1.e-12)[0]): # 極めて小さい固有値は削除する
		lambdas = np.delete(lambdas, i, 0)
		P = np.delete(P, i, 1)
	Atilda = la.inv(np.sqrt(np.diag(lambdas))) @ _H(P) # 球面化行列
	X_whiten = Atilda @ X_center
	if _assert:
		assert np.allclose(np.cov(X_whiten), np.eye(X_whiten.shape[0]), atol=1.e-10) # 無相関化を確認（単位行列）

	beta = 2.001953125
	# ICAに使用する関数gとその微分g2（ここではgは４次キュムラント）
	g = lambda bx : beta + np.tanh(bx) 
	g2 = lambda bx : 1+np.tanh(bx)
	g = lambda bx : bx**3
	g2 = lambda bx : 3*(bx**2)

	I = X_whiten.shape[0]
	B = np.array([[(np.random.rand()-0.5)+(np.random.rand()-0.5)*1j for i in range(I)] for j in range(I) ], dtype=np.complex) # X_whitenからYへの復元行列

	# Bを直交行列かつ列ベクトルが大きさ１となるように規格化
	for i in range(I):
		if i > 0:
			B[:,i] = B[:,i] - B[:,:i] @ _H(B[:,:i]) @ B[:,i] # 直交空間に射影
		B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化

	# Bの決定(Y = B.T @ X_whiten)
	for i in range(I):
		for j in range(1000):
			prevBi = B[:,i].copy()
			BiH = _H(B[:,i])
			result = []
			for x in X_whiten.T:
				BiHx = BiH@x
				BiHx2 = abs(BiHx)**2
				row = x*np.conjugate(BiHx)*g(BiHx2) - (g(BiHx2)+BiHx2*g2(BiHx2))*B[:,i]
				result.append(row)
			B[:,i] = np.average(result, axis=0) # 不動点法
			B[:,i] = B[:,i] - B[:,:i] @ _H(B[:,:i]) @ B[:,i] # 直交空間に射影
			B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化
			print(prevBi @ B[:,i])
			if 1.0 - 1.e-8 < abs(prevBi @ B[:,i]) < 1.0 + 1.e-8: # （内積1 <=> ほとんど変更がなければ）
				break
		else:
			assert False
	if _assert:
		assert np.allclose(B @ _H(B), np.eye(B.shape[0]), atol=1.e-10) # Bが直交行列となっていることを検証

	Y = _H(B) @ X_whiten

	return CFastICAResult(Y)

# ica = FastICA(n_components=SERIES, random_state=0)
# _Y = ica.fit_transform(X.T).T * 8

class EASI:
	def __init__(self, size: int, mu=0.001953125, g=lambda x:-np.tanh(x)):
		self.B = np.array([[np.random.rand()-0.5 for i in range(size)] for j in range(size) ]) # 復元行列
		self._g = g # 更新関数
		self._mu = mu # 更新時パラメータ
		self._size = size # 観測点数
	
	"""
	新しく観測したxを更新します
	x: ndarray (self.size長ベクトル)

	returns: 復元ベクトル
	"""
	def update(self, x: np.ndarray) -> np.ndarray:
		y = np.array([self.B @ x]).T
		V = y @ y.T - np.eye(self._size) + self._g(y) @ y.T - y @ self._g(y).T
		self.B = self.B - self._mu * V @ self.B
		return y[:,0]

@dataclasses.dataclass
class EASIResult:
	# Y represents obtained independent data.
	# 
	# EX:
	#   [[y_0(0), y_0(1), y_0(2)]
	#    [y_1(0), y_1(1), y_1(2)]]
	#   s.t. y_point(time)
	Y: np.ndarray

def BatchEASI(X: np.ndarray):
	signals = X.shape[0]
	easi = EASI(signals)
	YT = []
	for x in X.T:
		y = easi.update(x)
		YT.append(y)
	Y = np.array(YT).T
	return EASIResult(Y=Y)


"""
チェビシェフ系列を生成（第一種）
deg: チェビシェフ多項式の次数
a0: 初期値
length: 系列の長さ
"""
def chebyt_samples(deg: int, a0: float, length: int) -> np.ndarray:
	result = [a0]
	for _ in range(length-1):
		a0 = eval_chebyt(deg, a0)
		result.append(a0)
	return np.array(result) 

"""
チェビシェフ系列を生成（第二種）
deg: チェビシェフ多項式の次数
a0: 初期値
length: 系列の長さ
"""
def chebyu_samples(deg: int, a0: float, length: int) -> np.ndarray:
	result = [a0]
	for _ in range(length-1):
		a0 = eval_chebyu(deg, a0)
		result.append(a0)
	return np.array(result) 

"""
ワイル系列を生成
return ndarray(dtype=complex)
https://www.jstage.jst.go.jp/article/transcom/advpub/0/advpub_2017EBP3139/_article/-char/ja/
"""
def weyl_samples(low_k: float, delta_k: float, length: int) -> np.ndarray:
	result = []
	for n in range(length):
		x_raw = n*low_k + delta_k
		x = x_raw - np.floor(x_raw)
		result.append(np.exp(2 * np.pi * 1j * x))
	return np.array(result)

"""
複素数系列を生成
https://www.jstage.jst.go.jp/article/japannctam/55/0/55_0_81/_pdf/-char/ja
n: 何倍角の系列か
"""
def const_powerd_samples(n: int, a_0: complex, length: int) -> np.ndarray:
	result = []
	result.append(a_0)
	for _ in range(length-1):
		a_0 = complex(eval_chebyt(n, a_0.real), eval_chebyu(n, a_0.imag))
		result.append(a_0)
	return np.array(result, dtype=complex)

""" 
-0.5~+0.5なる混合行列を作成
size: 正方行列のサイズ
"""
def random_matrix(size: int) -> np.ndarray:
	return np.array([[np.random.rand()-0.5 for i in range(size)] for j in range(size) ])

"""
2つの行ごとの内積を計算し、行列にまとめます
"""
def correlation(P: np.ndarray) -> np.ndarray:
	res = np.eye(P.shape[0], dtype=P.dtype)
	for i in range(P.shape[0]):
		for j in range(P.shape[0]):
			res[i][j] = (P[i]@P[j]) / P.shape[1]
	return res

"""
implementation of (8) in
https://www.jstage.jst.go.jp/article/transcom/advpub/0/advpub_2017EBP3139/_article/-char/ja/
"""
def _matrix_c(P: np.ndarray, l: int) -> np.ndarray:
	N = int(P.shape[1]/2)
	result = []
	for i in range(P.shape[0]):
		row = []
		for k in range(P.shape[0]):
			if abs(l) >= N:
				row.append(0)
			elif l >= 0:
				_sum = 0
				for n in range(N-1):
					_sum += np.conjugate(P[i, n+l]) * P[k, n]
				row.append(_sum)
			else:
				_sum = 0
				for n in range(N+1):
					_sum += np.conjugate(P[i, n]) * P[k, n-l]
				row.append(_sum)
		result.append(row)
	return np.array(result)

"""
implementation of (6) in
https://www.jstage.jst.go.jp/article/transcom/advpub/0/advpub_2017EBP3139/_article/-char/ja/
"""
def periodic_correlation(P: np.ndarray, l: int):
	N = int(P.shape[1]/2)
	return _matrix_c(P, l) + _matrix_c(P, l-N)

"""
implementation of (7) in
https://www.jstage.jst.go.jp/article/transcom/advpub/0/advpub_2017EBP3139/_article/-char/ja/
"""
def odd_periodic_correlation(P: np.ndarray, l: int):
	N = int(P.shape[1]/2)
	return _matrix_c(P, l) - _matrix_c(P, l-N)
