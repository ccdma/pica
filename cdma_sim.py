import lib.ica as ica
import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.njit
def cdma():
    bits = ica.random_bits((3, 4))
    print(bits)

cdma()