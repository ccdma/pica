"""
原始根符号の相関特性(qを変えた場合)
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

p = 1019

print((p-1)/2)
print(lb.is_prime(int((p-1)/2)))

q_start = 2
q_range = range(q_start, q_start+1000)

q_list = []
for _q in q_range:
	if lb.is_primitive_root(p, _q):
		q_list.append(_q)

cor = []

for q in q_list[1:]:
	X1 = lb.primitive_root_code(p, 2)
	X2 = lb.primitive_root_code(p, q, q)

	X1_1 = np.append(1, X1)
	X2_1 = np.append(1, X2)
	cor.append(np.vdot(X1_1, X2_1))

plt.plot(np.abs(np.array(cor)/p))
plt.show()


