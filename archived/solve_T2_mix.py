"""
任意の配分におけるT2による2信号の分離を解析的に行う 
"""
import sympy as sp
import lb

def T2(x):
	return 2*x**2 - 1

alpha = sp.Symbol("\\alpha")
beta = sp.Symbol("\\beta")

x_0 = sp.Symbol("x_n")
y_0 = sp.Symbol("y_n")

m_0 = sp.Symbol("m_n")
# m_1 = sp.Symbol("m_{n+1}")
p_1 = sp.Symbol("p_{n+1}") # p_1 = (m_1 + alpha + beta)/2 として置き換え

solved = sp.solve((
	alpha * x_0**2 + beta * y_0**2 - p_1,
	alpha * x_0 + beta * y_0 - m_0
), (x_0, y_0))

x_val = lb.chebyt_code(2, 0.1, 2)
y_val = lb.chebyt_code(2, 0.2, 2)

alpha_val = 1
beta_val = 2
m_0_val = alpha_val*x_val[0]+beta_val*y_val[0]
m_1_val = alpha_val*x_val[1]+beta_val*y_val[1]
vars = [
	(m_0, m_0_val),
	(p_1, (m_1_val+alpha+beta)/2),
	(alpha, alpha_val),
	(beta, beta_val),
]
print(x_val)
print(y_val)
print(solved[0][0].subs(vars))
print(solved[0][1].subs(vars))


# print(sp.latex(solved))

pass

# display()