import lb
import numpy as np
import matplotlib.pyplot as plt

N = 15
K = 3

snrs = np.linspace(1.0, 5.0, 20)


ber = lb.ber(np.power(10, snrs/10))

ber = np.array(ber)
plt.plot(snrs, ber)
plt.yscale('log')
plt.show()