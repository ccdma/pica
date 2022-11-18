"""
k: N
θn = q^n / p
Ψn = q'^n / p'
Xn = exp(-2j*π*θn*k)*exp(-2j*π*Ψn*k)

1-x^2 = (1-x)(1+x)
原始根異なるものp,q色々変えてみる
p1,p2を大きくするとrsaっぽく作れる
符号数を増やせる、符号長をかんたんに長くできる
3つ以上でもやってみる
ビット同期でない場合の相関特性
"""
import lb
import numpy as np
import matplotlib.pyplot as plt
import math

pq_list = [(13, 2), (19, 2)]
code_len = math.lcm(*map(lambda pq: pq[0], pq_list))

code_1 = lb.mixed_primitive_root_code(pq_list, 1)

corr_list = []
for k in range(2, code_len):
    corr_list.append(np.vdot(lb.mixed_primitive_root_code(pq_list, k), code_1) / code_len)

# plt.plot(np.abs(corr_list))
# plt.title(f"(p, q)={pq_list}: correlation of X(k1=1) and X(k2)")
# plt.xlabel("k2")
# plt.ylabel("correlation")
# plt.tight_layout()
# plt.show()

plt.scatter(code_1.real, code_1.imag, s=1)
plt.plot(code_1.real, code_1.imag, lw=0.2)
plt.title(f"IQ plot of (p, q)={pq_list}")
plt.gca().set_aspect('equal','datalim')
plt.show()

