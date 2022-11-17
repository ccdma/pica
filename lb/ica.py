import numpy.linalg as la
import numpy as np
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
	W: np.ndarray

"""
X: represents observed data
[[x1(0), x1(0.1), x1(0.2)],[x2(0), x2(0.1), x2(0.2)]]のような感じ、右に行くにつれて時間が経過する
EX) 
   [[x_0(0), x_0(1), x_0(2)]
	[x_1(0), x_1(1), x_1(2)]]
	s.t. x_point(time) 
"""
def fast_ica(X: np.ndarray, _assert: bool=True) -> FastICAResult:
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

	W = B.T @ Atilda

	return FastICAResult(Y=Y, X_whiten=X_whiten, X_center=X_center, W=W)


"""
エルミート転置をする（随伴行列を求める）
"""
def hermite(P: np.ndarray):
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
def cfast_ica(X: np.ndarray, _assert: bool=True) -> CFastICAResult:
	SAMPLE, SERIES = X.shape # (観測点数, 観測時間数)

	# 中心化を行う（観測点ごとの平均であることに注意）
	mean = np.mean(X,axis=1)
	X_center = X - np.array([ np.full(SERIES, ave) for ave in mean ]) 

	# 固有値分解により、白色化されたX_whitenを計算する
	lambdas, P = la.eig(np.cov(X_center))
	if _assert:
		assert np.allclose(np.cov(X_center), P @ np.diag(lambdas) @ hermite(P)) # 固有値分解の検証
	for i in reversed(np.where(lambdas < 1.e-12)[0]): # 極めて小さい固有値は削除する
		lambdas = np.delete(lambdas, i, 0)
		P = np.delete(P, i, 1)
	Atilda = la.inv(np.sqrt(np.diag(lambdas))) @ hermite(P) # 球面化行列
	X_whiten = Atilda @ X_center
	if _assert:
		assert np.allclose(np.cov(X_whiten), np.eye(X_whiten.shape[0]), atol=1.e-10) # 無相関化を確認（単位行列）

	# ICAに使用する関数gとその微分g2（ここではgは４次キュムラント）
	g = lambda bx : np.tanh(bx) 
	g2 = lambda bx : 1+np.tanh(bx)

	I = X_whiten.shape[0]
	B = np.array([[(np.random.rand()-0.5)+(np.random.rand()-0.5)*1j for i in range(I)] for j in range(I) ], dtype=np.complex) # X_whitenからYへの復元行列

	# Bを直交行列かつ列ベクトルが大きさ１となるように規格化
	for i in range(I):
		if i > 0:
			B[:,i] = B[:,i] - B[:,:i] @ hermite(B[:,:i]) @ B[:,i] # 直交空間に射影
		B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化

	# Bの決定(Y = B.T @ X_whiten)
	for i in range(I):
		for j in range(1000):
			prevBi = B[:,i].copy()
			BiH = hermite(B[:,i])
			result = []
			for x in X_whiten.T:
				BiHx = np.vdot(BiH,x)
				BiHx2 = abs(BiHx)**2
				row = x*np.conjugate(BiHx)*g(BiHx2) - (g(BiHx2)+BiHx2*g2(BiHx2))*B[:,i]
				result.append(row)
			B[:,i] = np.average(result, axis=0) # 不動点法
			B[:,i] = B[:,i] - B[:,:i] @ hermite(B[:,:i]) @ B[:,i] # 直交空間に射影
			B[:,i] = B[:,i] / la.norm(B[:,i], ord=2) # L2ノルムで規格化
			# print(abs(prevBi @ B[:,i]))
			if 1.0 - 1.e-4 < abs(prevBi @ B[:,i]) < 1.0 + 1.e-4: # （内積1 <=> ほとんど変更がなければ）
				break
		else:
			assert False
	if _assert:
		assert np.allclose(B @ hermite(B), np.eye(B.shape[0]), atol=1.e-10) # Bが直交行列となっていることを検証

	Y = hermite(B) @ X_whiten
	
	return CFastICAResult(Y)

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

def batch_easi(X: np.ndarray):
	signals = X.shape[0]
	easi = EASI(signals)
	YT = []
	for x in X.T:
		y = easi.update(x)
		YT.append(y)
	Y = np.array(YT).T
	return EASIResult(Y=Y)

"""
S ≒ P.T*Y なる循環行列Pを生成します
A: 混合行列
W: 復元行列
"""
def estimate_circulant_matrix(A, W):
	G = W @ A
	P = np.zeros(G.shape)
	for i, g in enumerate(G):
		mi = np.argmax(np.abs(g))
		if g[mi] > 0:
			P[i,mi] = 1
		else:
			P[i,mi] = -1
	return P
