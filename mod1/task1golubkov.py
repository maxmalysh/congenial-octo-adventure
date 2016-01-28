import numpy as np
from typing import List
from enum import Enum
import scipy.linalg


class PivotMode(Enum):
    BY_ROW = 0
    BY_COLUMN = 1
    BY_MATRIX = 2


def pivot_by_row(A: np.matrix, k: int, row_permutation: List[float]):
    mat_size = A.shape[0]
    row_with_max_leading_elem = k # First row of leading submatrix

    # Find the row with largest leading element
    for i in range(k+1, mat_size):
        if abs(A[row_permutation[i], k]) > abs(A[row_permutation[row_with_max_leading_elem], k]):
            row_with_max_leading_elem = i

    if row_with_max_leading_elem != k:
        # swap rows k, row_with_max_leading_elem
        row_permutation[row_permutation[k]] = row_with_max_leading_elem
        row_permutation[row_permutation[row_with_max_leading_elem]] = k


def pivot_by_column(A: np.matrix, k: int, column_permutation: List[float]):
    mat_size = A.shape[0]
    column_with_max_leading_elem = k

    for i in range(k + 1, mat_size):
        if abs(A[k, column_permutation[i]]) > abs(A[k, column_permutation[column_with_max_leading_elem]]):
            column_with_max_leading_elem = i

    if column_with_max_leading_elem != k:
        column_permutation[column_permutation[k]] = column_with_max_leading_elem
        column_permutation[column_permutation[column_with_max_leading_elem]] = k


def pivot_by_matrix(A: np.matrix, k: int, row_permutation: List[float], column_permutation: List[float]):
    mat_size = A.shape[0]
    row_with_max_leading_elem = k
    column_with_max_leading_elem = k

    for i in range(k + 1, mat_size):
        for j in range(k + 1, mat_size):
            if abs(A[row_permutation[i], column_permutation[k]]) > abs(
                    A[row_permutation[row_with_max_leading_elem], column_permutation[k]]):
                row_with_max_leading_elem = i
                column_with_max_leading_elem = j

    row_permutation[row_permutation[k]] = row_with_max_leading_elem
    row_permutation[row_permutation[row_with_max_leading_elem]] = k
    column_permutation[column_permutation[k]] = column_with_max_leading_elem
    column_permutation[column_permutation[column_with_max_leading_elem]] = k


def pivot(A: np.matrix, k: int, mode: PivotMode, row_permutation: List[float], column_permutation: List[float]):
    if mode == PivotMode.BY_ROW:
        pivot_by_row(A, k, row_permutation)
    elif mode == PivotMode.BY_COLUMN:
        pivot_by_column(A, k, column_permutation)
    elif mode == PivotMode.BY_MATRIX:
        pivot_by_matrix(A, k, row_permutation, column_permutation)


'''
Decompositions:
- PLU (pivoting by row)
- LUP (pivoting by column)
- PLUP' (pivoting by submatrix)

where
L is lower triangular,
U is upper unitriangular
(like in the task).

Returns: (A, row_permutation, column_permutation) tuple.
A - all-in-one matrix for L, U.
row_permutation - permutation array for rows. Example: [0, 1, 2] is identity, [1, 0, 2] is rows 0, 1 swapped.
column_permutation - permutation array for columns.

For PLU, returned column_permutation is always identity, likewise for LUP row_permutation is always identity.
'''

def lu_decompose_pivoting(A: np.matrix, mode: PivotMode):
    n = A.shape[0]
    row_perm = [i for i in range(n)]  # Row permutation. Identity permutation at first.
    column_perm = [i for i in range(n)]  # And column permutation.

    for k in range(0, n): # k - Gaussian step number
        pivot(A, k, mode, row_perm, column_perm)
        # Divide the first row of current Gaussian step's submatrix by its leading element. (LecV.pdf p. 18, point 1).
        # The leading element itself is preserved (LecV.pdf p. 21).
        for j in range(k + 1, n):
            A[row_perm[k], column_perm[j]] /= A[row_perm[k], column_perm[k]]
        # From each row below the first one in current Gaussian step's submatrix,
        # subtract the first row multiplied by the first element of the row we subtract from. (LecV.pdf p. 18, point 2).
        # First elements are preserved. (LecV.pdf p. 21).
        for i in range(k+1, n): # i - row that we will subtract from
            for j in range(k + 1, n):  # j - column index
                A[row_perm[i], column_perm[j]] -= A[row_perm[k], column_perm[j]] * A[row_perm[i], column_perm[k]]

    return A, row_perm, column_perm


'''
Extract L and U from all-in-one matrix.

Returns: (L, U) tuple.
'''

def lu_extract(LU: np.matrix, row_permutation: List[float], column_permutation: List[float]):
    mat_size = LU.shape[0]
    L = np.zeros((mat_size, mat_size))
    U = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        for j in range(mat_size):
            if j > i:
                U[i, j] = LU[row_permutation[i], column_permutation[j]]
            elif j < i:
                L[i, j] = LU[row_permutation[i], column_permutation[j]]
            elif j == i:
                L[i, j] = LU[row_permutation[i], column_permutation[j]]
                U[i, j] = 1
    return L, U


# @vector - permutation vector as a list, e.g. [2,0,1]
# @row – boolean determining whether vector is shuffling rows or columns
def perm_vector_to_matrix(vector, row=True):
    n = len(vector)
    matrix = np.zeros(shape=(n,n))

    for i in range(0, n):
        if row:
            matrix[vector[i], i] = 1
        else:
            matrix[i, vector[i]] = 1

    return matrix

def PLU_decomposition(A):
    mode = PivotMode.BY_ROW
    lu_in_one, P, P_ = lu_decompose_pivoting(A.astype(np.float), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, U

def LUP_decomposition(A):
    mode = PivotMode.BY_COLUMN
    lu_in_one, P, P_ = lu_decompose_pivoting(A.astype(np.float), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return L, U, perm_vector_to_matrix(P_, row=False)

def PLUP_decomposition(A):
    mode = PivotMode.BY_MATRIX
    lu_in_one, P, P_ = lu_decompose_pivoting(A.astype(np.float), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, U, perm_vector_to_matrix(P_, row=False)
