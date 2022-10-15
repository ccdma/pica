from pandas import qcut
from lib.ica import *
import matplotlib.pyplot as plt

p = 1019
q = 2

print((p-1)/2)
print(is_prime(int((p-1)/2)))
X1 = primitive_root_code(p, q, 1)
X2 = primitive_root_code(p, q, 1)

cor = []
print(np.vdot(np.append(1, X1), np.append(1, X2)))
print(np.vdot(X1, X2))

for i in range(p):
	cor.append(np.vdot(X1, np.roll(X2, i)))

plt.plot(np.array(cor).real/p)
plt.show()


