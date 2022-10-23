from lib.ica import *
import matplotlib.pyplot as plt

p = 1019
q = 2

print((p-1)/2)
print(is_prime(int((p-1)/2)))

cor = []

for i in range(p-2):
	X1 = primitive_root_code(p, q, 1)
	X2 = primitive_root_code(p, q, i+2)

	X1_1 = np.append(1, X1)
	X2_1 = np.append(1, X2)
	cor.append(np.vdot(X1_1, X2_1))

plt.scatter(np.array(cor).real/p, np.array(cor).imag/p)
plt.show()


