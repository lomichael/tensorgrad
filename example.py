import sys
print(sys.path)

from tensorgrad.tensor import Tensor 

a = Tensor([2.0])
b = Tensor([3.0])
c = Tensor([4.0])

d = a * b + c

d.backward()

print(a.grad)
print(b.grad)
print(c.grad)
