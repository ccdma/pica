"""
実数信号においてシンボリックダイナミクスで挙動を確認
"""
import lb
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def odd_or_even(i: int):
    if i%2 == 0:
        return 1
    else:
        return -1

def cte(C: np.ndarray):
    size = C.shape[0]
    _results = []
    for i in range(size):
        absvec = np.abs(C[i])
        max_k = np.max(absvec)
        _sum = np.sum(absvec)
        _results.append(_sum/max_k-1)
    for i in range(size):
        absvec = np.abs(C[:, i])
        max_k = np.max(absvec)
        _sum = np.sum(absvec)
        _results.append(_sum/max_k-1)
    return sum(_results)

SIGNALS = 3
SAMPLINGS = 1000

def test(signals: int, samplings: int, norm_scale: float):

    B = np.array([[ odd_or_even(np.random.randint(0, 10)) for i in range(samplings) ] for _ in range(signals)])
    S = np.array([ lb.chebyt_code(2, 0.1+i/10, samplings) for i in range(signals)])

    T = S * B

    print(lb.correlation(T))

    A = lb.random_matrix(signals)

    X = A @ T + np.random.normal(0.0, norm_scale, (signals, samplings))

    res = lb.fast_ica(X, _assert=False)

    Y = res.Y
    P = lb.estimate_circulant_matrix(A, res.W)

    S2 = P.T @ Y

    RB = np.sign(S2*S)

    # BER
    cber = np.abs(RB - B).mean()/2
    # CTE
    ccte = cte(res.W @ A)
    print(cber, ccte)

    return (cber, ccte)

scales = np.linspace(0.1, 0.2, 1)
cber_avgs = []
ccte_avgs = []
for scale in scales:
    cber_list = []
    ccte_list = []
    for _ in range(1):
        cber, ccte = test(SIGNALS, SAMPLINGS, scale)
        cber_list.append(cber)
        ccte_list.append(ccte)
    cber_avgs.append(np.mean(cber_list))
    ccte_avgs.append(np.mean(ccte_list))

plt.plot(scales, ccte_avgs)
plt.show()