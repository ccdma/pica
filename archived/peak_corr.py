"""
ピークを測定
"""
import lb
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
np.set_printoptions(suppress=True, linewidth=1000)

SAMPLINGS = 10000
SIGNALS = 2

s = lb.const_power_code(2, 0.1, SAMPLINGS)
s = s * np.tile(np.exp(np.linspace(0, 2*np.pi, 100)*(-1j)), 500)[:SAMPLINGS]
# s = s[:500]
# plt.plot(s.real, s.imag, lw=0.3)
# plt.show()
# exit()
s = s.real

coll = []
for slices in range(-100, 100):
	base = 1000
	length = 300
	s1 = s[base:base+length]
	s2 = s[base+slices:base+slices+length]
	coll.append((s1@s2)/length)
plt.plot(coll)
plt.show()

# rr = FastICA(X.real, _assert=False)
# ri = FastICA(X.imag, _assert=False) 

p = 1021
for i in range(p):
	if lb.is_primitive_root(p, i+1):
		print(i+1)