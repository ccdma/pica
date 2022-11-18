"""
原始根符号の相関特性(kを変えた場合)
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

p = 1019
q = 2

print((p-1)/2)
print(lb.is_prime(int((p-1)/2)))

exact_correlation = []

for i in range(p-2):
	X1_1 = lb.primitive_root_code(p, q, 1)
	X2_1 = lb.primitive_root_code(p, q, i+2)

	exact_correlation.append(np.vdot(X1_1, X2_1))

plt.plot(np.abs(np.array(exact_correlation)/p))
plt.title("correlation of X(k1=1) and X(k2)")
plt.xlabel("k2")
plt.ylabel("correlation")
plt.show()


