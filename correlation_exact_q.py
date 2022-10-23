from lib.ica import *
import matplotlib.pyplot as plt

p = 1019

print((p-1)/2)
print(is_prime(int((p-1)/2)))

q_start = 2
q_range = range(q_start, q_start+1000)

q_list = []
for _q in q_range:
	if is_primitive_root(p, _q):
		q_list.append(_q)

cor = []

for q in q_list[1:]:
	X1 = primitive_root_code(p, 2)
	X2 = primitive_root_code(p, q)

	X1_1 = np.append(1, X1)
	X2_1 = np.append(1, X2)
	cor.append(np.vdot(X1_1, X2_1))

plt.plot(np.abs(np.array(cor)/p))
plt.show()


