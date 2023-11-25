import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a, a.shape)
b = np.array([[7, 8, 1, 1], [9, 10, 1, 1], [11, 12, 1, 1]])
print(b, b.shape)
c = a @ b
print(c, c.shape)
