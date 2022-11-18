"""
任意の配分におけるT2による2信号の分離を解析的に行う 
"""
import sympy as sp
import lb

def T2(x, n=1):
	for _ in range(n):
		x *= 2*x**2 - 1
	return x

alpha_1 = sp.Symbol("\\alpha_1")
alpha_2 = sp.Symbol("\\alpha_2")

x1_0 = sp.Symbol("X^1_n")
x2_0 = sp.Symbol("X^2_n")

m_0 = sp.Symbol("M_n")
m_1 = sp.Symbol("M_{n+1}")
m_2 = sp.Symbol("M_{n+2}")
m_3 = sp.Symbol("M_{n+3}")

# m_1 = sp.Symbol("m_{n+1}")
# p_1 = sp.Symbol("p_{n+1}") # p_1 = (m_1 + alpha + beta)/2 として置き換え

solved = sp.solve(
	[alpha_1 * T2(x1_0, i) + alpha_2 * T2(x2_0, i) - m for i, m in enumerate([m_0, m_1, m_2, m_3])],
	(x1_0, x2_0, alpha_1, alpha_2)
)

x1_val = lb.chebyt_code(2, 0.1, 4)
x2_val = lb.chebyt_code(2, 0.2, 4)

alpha1_val = 1
alpha2_val = 2
m_val = [alpha1_val*x1_val[i]+alpha2_val*x2_val[i] for i in range(4)]

vars = [
	(m_0, m_val[0]),
	(m_1, m_val[1]),
	(m_2, m_val[2]),
	(m_3, m_val[3]),
]
print(x1_val)
print(x2_val)
print(solved[0][0].subs(vars))
print(solved[0][1].subs(vars))


print(sp.latex(solved))

pass

# display()