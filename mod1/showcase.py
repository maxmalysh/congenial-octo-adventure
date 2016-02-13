import numpy as np
from functools import reduce

from task1 import PLU_decomposition, LUP_decomposition, PLUP_decomposition
from task1 import lpu_decompose, lpl_decompose
from task2 import plup_solve

#
# Generating test cases
#
from utils import MatrixBuilder

diag_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype = np.float)

vector1 = np.array([
    10.0,
    20.0,
    30.0,
#    100.0,
], dtype = np.float)

hilbert_matrix = MatrixBuilder(3).hilbert().gen()

#
# Checking decompositions
#

input_matrix = hilbert_matrix
input_vector = vector1

#  result = PLU_decomposition(input_matrix)
#  result = LUP_decomposition(input_matrix)
#  result = PLUP_decomposition(input_matrix)
result = lpu_decompose(input_matrix)
# result = lpl_decompose(input_matrix)

print("Input:")
print(input_matrix)

print("Output:")
for x in result:
    print(x)

print("Check:")
print(reduce(lambda M, LM: M @ LM, result))


#
# Solving a system
#
A = input_matrix
b = input_matrix
print("Solving system with the following right-hand-side vector:")
print(b)

print("Answer:")
x = plup_solve(A, b)

print(x)
print("Residual:")
print(b - A.dot(x))
