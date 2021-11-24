from ica import *

p = 1024
for i in range(p):
	if is_primitive_root(p, i+1):
		print(i+1)