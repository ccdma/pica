import lb, itertools, random, dataclasses
import numpy as np
import matplotlib.pyplot as plt

N = 21
K = 3
stddevs = np.linspace(0.1, 1.0, 30)

ber = []
for stddev in np.linspace(0.1, 1.0, 30):
	ber.append(lb.cdma_ber(N, stddev, K))

ber = np.array(ber)
plt.plot(stddevs, ber)
plt.show()