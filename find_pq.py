"""
原始根を探索する
"""
import lb, typing

def linext(start: int, len: int = 1) -> typing.Iterator[int]:
	return range(start, start+len)

result = lb.find_pq(
	linext(3, 40),
	linext(2, 5)
)
print(result)