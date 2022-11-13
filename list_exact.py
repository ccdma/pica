from lib.ica import *

p_start = 2
p_range = range(p_start, p_start+100)

q_start = 2
q_range = range(q_start, q_start+1)

result = []
for q in q_range:
	for p in p_range:
		if is_primitive_root(p, q):
			result.append((p, q))

print(result)