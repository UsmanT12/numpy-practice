import numpy as np

#Following tutorial of numpy from freecodecamp

a = np.array([1, 2, 3])
b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])

print(a)
print("a type: ", a.dtype)
print("a shape: ", a.shape)
print("a dimension: ", a.ndim)
print("a item size: ", a.itemsize, '\n')

print(b)
print("b type: ", b.dtype)
print("b shape: ", b.shape)
print("b dimension: ", b.ndim)
print("b item size: ", b.itemsize, '\n')

print(b.shape[0])

c = np.zeros((4, 4))
print(c)