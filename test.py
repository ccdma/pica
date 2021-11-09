from ica import *

SAMPLINGS = 10000
SIGNALS = 4

for i in range(10):
	S = np.array([chebyt_samples(j+2, 0.1, SAMPLINGS) for j in range(SIGNALS)])
	X = random_matrix(SIGNALS) @ S
	Y = FastICA(X, _assert=False)

