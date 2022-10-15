from lib.ica import *
import matplotlib.pyplot as plt

p = 1019
q = 2

print((p-1)/2)
print(is_prime(int((p-1)/2)))
code = primitive_root_code(p, q)

cor = []
for i in range(p):
	X1 = np.roll(code, 0)
	X2 = np.roll(code, i)
	cor.append(X1@X2)

plt.plot(np.array(cor).real/p)
plt.show()


