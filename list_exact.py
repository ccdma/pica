from lib.ica import *

q = 2

p_start = 1000
p_range = range(p_start, p_start+1000)

result = []
for p in p_range:
	if is_primitive_root(p, q):
		result.append((p, q))

print(result)