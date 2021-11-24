from ica import *

for i in range(1020, 1030):
	if is_prime(i):
		print(i)

p = 1021
for i in range(p):
	if is_primitive_root(p, i+1):
		print(i+1)