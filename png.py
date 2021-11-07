from pica.ica import const_powerd_samples
import matplotlib.pyplot as plt
import numpy as np
from pica.ica import *

SIGNALS = 3
SAMPLINGS = 1000

SC = np.array([ const_powerd_samples(i+2, np.pi/12, 1024) for i in range(SIGNALS)]) 

S = []
for s in S:
	S.append(s.real)
	S.append(s.imag)

A = random_matrix(SIGNALS)

X = A @ S

res = FastICA(X, _assert=False)

Y = res.Y

plt.scatter(Y[1].real, Y[1].imag)
plt.show()
