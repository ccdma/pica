"""
原始根を探索する
"""
import lb

p_start = 2
q_start = 2

result = lb.find_pq(
	range(p_start, p_start+300),
	range(q_start, q_start+4)
)
print(result)