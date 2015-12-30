import numpy as np
from typing import List
import scipy.linalg

test_matrices = [

    np.matrix([
        [7, 3, -1, 2],
        [3, 8, 1, -4],
        [-1, 1, 4, -1],
        [2, -4, -1, 6],
    ]),

    np.matrix([
        [0,  1, 0],
        [-8, 8, 1],
        [2, -2, 0],
    ]),

]

def pivot_by_row(A: np.matrix, k: int, row_permutation: List[float]):
    mat_size = A.shape[0]
    max_row = k # first fow of leading submatrix
    for i in range(k+1, mat_size):
        if abs(A[row_permutation[i], k]) > abs(A[row_permutation[max_row], k]):
            max_row = i

    # swap rows k, max_row
    row_permutation[row_permutation[k]] = max_row
    row_permutation[row_permutation[max_row]] = k


def plu_decompose(A: np.matrix) -> np.matrix:
    mat_size = A.shape[0]
    row_permutation = [i for i in range(mat_size)] # identity permutation, row_permutation[i] = i

    for k in range(mat_size): # k: the number of Gauss steps
        pivot_by_row(A, k, row_permutation)
        print(str(k) + " " + str(row_permutation))
        leading_elem = A[row_permutation[k], k] # A_kk is leading element
        for i in range(k+1, mat_size): # i: row index, the row we subtract scaled first row from
            scale_factor = A[row_permutation[i], k] / leading_elem
            for j in range(k+1, mat_size): # j: column index
                # j starts from second column,
                #  we do not zero first column, as in usual Gauss elimination
                A[row_permutation[i], j] -= A[row_permutation[k], j] * scale_factor

    return (A, row_permutation)

def lu_extract(LU: np.matrix, row_permutation: List[float]):
    mat_size = A.shape[0]
    L = np.zeros((mat_size, mat_size))
    U = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        for j in range(mat_size):
            if j > i:
                U[i, j] = A[row_permutation[i], j]
            elif j < i:
                L[i, j] = A[row_permutation[i], j]
            elif j == i:
                U[i, j] = A[row_permutation[i], j]
                L[i, j] = 1
                pass

    return (L, U)


A = test_matrices[1]

plu = plu_decompose(A)
L, U = lu_extract(plu[0], plu[1])
print (L)
print (U)
print (L@U)
print ()



P, L, U = scipy.linalg.lu(A)
print (P)
print (L)
print (U)
print (L@U)