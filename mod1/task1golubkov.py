import numpy as np
from typing import List
import scipy.linalg

test_matrices = [

    np.matrix([
        [7, 3, -1, 2],
        [3, 8, 1, -4],
        [-1, 1, 4, -1],
        [2, -4, -1, 6],
    ], dtype=float),

    np.matrix([
        [2, -2, 0],
        [0,  1, 0],
        [-8, 8, 1],
    ], dtype=float),

]

def pivot_by_row(A: np.matrix, k: int, row_permutation: List[float]):
    mat_size = A.shape[0]
    row_with_max_leading_elem = k # First row of leading submatrix

    # Find the row with largest leading element
    for i in range(k+1, mat_size):
        if abs(A[row_permutation[i], k]) >= abs(A[row_permutation[row_with_max_leading_elem], k]):
            row_with_max_leading_elem = i

    if row_with_max_leading_elem != k:
        # swap rows k, row_with_max_leading_elem
        row_permutation[row_permutation[k]] = row_with_max_leading_elem
        row_permutation[row_permutation[row_with_max_leading_elem]] = k

# Returns: (A, row_permutation) tuple.
# A - all-in-one matrix: lower unitriangle L, upper tringle U.
# row_permutation - permutation array for rows. Example: [0, 1, 2] is identity, [1, 0, 2] is rows 0, 1 swapped.
def decompose_pivoting_by_row(A: np.matrix):
    n = A.shape[0]
    row_permutation = [i for i in range(n)] # identity permutation, row_permutation[i] = i

    # On this kij decomposition and others:
    # http://www-users.cselabs.umn.edu/classes/Spring-2014/csci8314/FILES/LecN6.pdf p. 6
    for k in range(0, n):
        pivot_by_row(A, k, row_permutation)
        for i in range(k+1, n):
            A[row_permutation[i], k] /= A[row_permutation[k], k]
            for j in range(k+1, n):
                A[row_permutation[i], j] -= A[row_permutation[i], k]*A[row_permutation[k], j]
    return A, row_permutation

# Returns: (L, U) tuple.
# Extract L and U from all-in-one matrix.
def lu_extract(LU: np.matrix, row_permutation: List[float]):
    mat_size = LU.shape[0]
    L = np.zeros((mat_size, mat_size))
    U = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        for j in range(mat_size):
            if j > i:
                U[i, j] = LU[row_permutation[i], j]
            elif j < i:
                L[i, j] = LU[row_permutation[i], j]
            elif j == i:
                U[i, j] = LU[row_permutation[i], j]
                L[i, j] = 1
    return L, U


A = test_matrices[1]

print(A)
print()

print("decompose_pivoting_by_row")
lu_in_one, P = decompose_pivoting_by_row(A.copy())
L, U = lu_extract(lu_in_one, P)
print("P", P)
print(P)
print("L")
print(L)
print("U")
print(U)
print("LU")
print(L@U)
print()

print("SCIPY")
P, L, U = scipy.linalg.lu(A.copy())
print("P")
print (P)
print("L")
print (L)
print("U")
print (U)
print("LU")
print (L@U)