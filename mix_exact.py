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
from lib.ica import *
import matplotlib.pyplot as plt
import math

def lcm(a: int, b: int):
    return a * b // math.gcd(a, b)

def make_code(p_a, q_a, p_b, q_b, k=1):
    lcm_ab = lcm(p_a, p_b)
    code_a = np.tile(np.append(1, primitive_root_code(p_a, q_a, k, False)), lcm_ab//p_a)
    code_b = np.tile(np.append(1, primitive_root_code(p_b, q_b, k, False)), lcm_ab//p_b)
    return code_a * code_b

p_a, q_a = 11, 2
p_b, q_b = 7, 3
lcm_ab = lcm(p_a, p_b)

code_1 = make_code(p_a, q_a, p_b, q_b, 1)
corr_list = []
for k in range(2, lcm_ab):
    corr_list.append(np.vdot(make_code(p_a, q_a, p_b, q_b, k), code_1) / lcm_ab)

# plt.plot(np.abs(corr_list))
# plt.title(f"(p, q)=({p_a},{q_a}),({p_b},{q_b}): correlation of X(k1=1) and X(k2)")
# plt.xlabel("k2")
# plt.ylabel("correlation")
# plt.tight_layout()
# plt.show()

plt.scatter(code_1.real, code_1.imag, s=3)
plt.plot(code_1.real, code_1.imag, lw=1)
plt.title(f"IQ plot of (p, q)=({p_a},{q_a}),({p_b},{q_b})")
plt.gca().set_aspect('equal','datalim')
plt.show()

