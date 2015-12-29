import scipy.linalg
import numpy as np
from pprint import pprint

# Делаем исходную матрицу A
A = scipy.array(
    [ [7, 3, -1, 2],
      [3, 8, 1, -4],
      [-1, 1, 4, -1],
      [2, -4, -1, 6] ]
)

A = scipy.array(
    [[0, 1, 0],
     [-8, 8, 1],
     [2,-2, 0]]
)

P, L, U = scipy.linalg.lu(A)

print("A:")
pprint(A)

print("P:")
pprint(P)

print("L:")
pprint(L)

print("U:")
pprint(U)




#
#
# https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy
# http://www.math.unm.edu/~loring/links/linear_s08/LU.pdf
#