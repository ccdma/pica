from lib.ica import *
import matplotlib.pyplot as plt

code = primitive_root_code(1019, 2)
code1 = chebyt_samples(2, 0.1, 1019)

res = []

for i in range(1019):
	res.append(code@primitive_root_code(1019, 2, i+1))

# code1 = chebyt_samples(2, 0.1, 1019)

# res = []

# for i in range(1019):
# 	res.append(code@np.roll(code, i))

plt.plot(np.array(res).real)
plt.show()