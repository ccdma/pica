"""
実数値の場合の相関をプロット
"""
from lib.ica import *
import matplotlib.pyplot as plt

correlation = []
code_len = 1019

X1 = const_powerd_samples(2, 0.1, code_len)
X2 = const_powerd_samples(2, 0.2, code_len)

for i in range(1, code_len):
	correlation.append(np.vdot(X1, np.roll(X2, i)))

plt.plot(np.abs(np.array(correlation)/code_len))
plt.title("correlation of const power code (inital_val=0.1,0.2)")
plt.xlabel("n = roll")
plt.ylabel("correlation")
plt.show()
