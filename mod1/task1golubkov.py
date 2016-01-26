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
PLU (pivoting by row), LUP (pivoting by column), PLUP' (pivoting by submatrix) decompositions.

Returns: (A, row_permutation, column_permutation) tuple.
A - all-in-one matrix: lower unitriangle L, upper tringle U.
row_permutation - permutation array for rows. Example: [0, 1, 2] is identity, [1, 0, 2] is rows 0, 1 swapped.
column_permutation - permutation array for columns.

For PLU, returned column_permutation is always identity, likewise row_permutation is always identity for LUP.
'''


def lu_decompose_pivoting(A: np.matrix, mode: PivotMode):
    n = A.shape[0]
    row_perm = [i for i in range(n)]  # Row permutation. Identity permutation at first.
    column_perm = [i for i in range(n)]  # And column permutation.

    # On this kij decomposition and others:
    # http://www-users.cselabs.umn.edu/classes/Spring-2014/csci8314/FILES/LecN6.pdf p. 6
    for k in range(0, n): # k - Gaussian step number
        pivot(A, k, mode, row_perm, column_perm)
        for i in range(k+1, n): # i - row that we will subtract from
            # Here we look at submatrix that corresponds to k-th Gaussian step.
            # In Gauss method, we would zero first element of each row below first one.
            # But here we store divider as first element of respective row, instead:
            divider = A[row_perm[i], column_perm[k]] / A[row_perm[k], column_perm[k]]
            A[row_perm[i], column_perm[k]] = divider
            # And the rest of elements in the row is subtracted from as usual:
            for j in range(k+1, n):
                A[row_perm[i], column_perm[j]] -= A[row_perm[k], column_perm[j]] * divider
    # Now in A we have these elements:
    # diagonal and above diagonal: result of usual Gaussian method,
    # below diagonal: all the dividers for each Gaussian step.
    # So we actually have complete information about all elementary
    # transformations we performed: dividers and row swaps that are stored in p.
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
                U[i, j] = LU[row_permutation[i], column_permutation[j]]
                L[i, j] = 1
    return L, U


def demo():
    test_matrices = [

        np.matrix([
            [7, 3, -1, 2],
            [3, 8, 1, -4],
            [-1, 1, 4, -1],
            [2, -4, -1, 6],
        ], dtype=float),

        np.matrix([
            [2, -2, 0],
            [0, 1, 0],
            [-8, 8, 1],
        ], dtype=float),
    ]

    A = test_matrices[1]
    mode = PivotMode.BY_MATRIX

    print(A)
    print()

    print("lu_decompose_pivoting: " + str(mode))
    lu_in_one, P, P_ = lu_decompose_pivoting(A.copy(), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    print("P")
    print(P)
    print("P'")
    print(P_)
    print("L")
    print(L)
    print("U")
    print(U)
    print("LU")
    print(L @ U)
    print()

    print("SCIPY")
    P, L, U = scipy.linalg.lu(A.copy())
    print("P")
    print(P)
    print("L")
    print(L)
    print("U")
    print(U)
    print("LU")
    print(L @ U)


if __name__ == "__main__":
    demo()
