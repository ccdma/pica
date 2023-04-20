import lb, itertools, random, dataclasses
import numpy as np
import matplotlib.pyplot as plt

N = 15
K = 3

snrs = np.linspace(1.0, 5.0, 20)

stddevs = np.linspace(0.5, 1.0, 30)

ber = []
for stddev in stddevs:
	ber.append(lb.cdma_ber(N, stddev, K))

ber = np.array(ber)
plt.plot(stddevs, ber)
plt.yscale('log')
plt.show()