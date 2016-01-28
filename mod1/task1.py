import numpy as np
from typing import List
from enum import Enum


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
LPU decomposition (modified Bruhat decomposition):

where
L is lower triangular,
U is upper unitriangular,
P is transposition matrix.
(like in the task).

# Not an in-place algorithm.
'''


def lpu_decompose(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    n = A.shape[0]
    L = np.identity(A.shape[0])
    U = np.identity(A.shape[0])

    # The algorithm: LecV.pdf p. 27.
    for k in range(0, n):  # k - Gaussian step number
        first_nonzero_elem_j = np.where(A[k] != 0)[1][
            0]  # Get tuple: (nonzero elements of row A[k], their indices). Get indices, get first index.
        first_nonzero_elem = A[k, first_nonzero_elem_j]

        # L, U start as identity matrices.
        # L accumulates inverses L^(-1)_k of Gaussian step transformation matrices applied to rows.
        # U accumulates inverses U^(-1)_k of Gaussian step transformation matrices applied to columns.
        # The matrices L_k, U_k that perform transformation done at k-th Gaussian step are provided in LecV.pdf p. 28.
        # Their inverses L^(-1)_k, U^(-1)_k are straightforward: invert diagonal elements,
        # multiply non-diagonal elements by -1 and divide by the diagonal element.

        # Scale the row
        for j in range(0, n):
            A[k, j] /= first_nonzero_elem

        # Store inverse row scale operation in L
        L[k, k] = first_nonzero_elem

        # Eliminate rows
        for s in range(k + 1, n):  # s - row to subtract from, to zero all elements below first_nonzero_elem.
            multiplier = A[s, first_nonzero_elem_j]
            for j in range(0, n):
                A[s, j] -= A[k, j] * multiplier
            # Store inverse row elimination operation in L
            L[s, k] = multiplier

        # Eliminate columns
        for t in range(first_nonzero_elem_j + 1,
                       n):  # t - column to subtract from, to zero all elements to the right of first_nonzero_elem.
            multiplier = A[k, t] / A[k, first_nonzero_elem_j]
            for i in range(0, n):
                A[i, t] -= A[i, first_nonzero_elem_j] * multiplier
            # Store inverse column elimination operation in U
            U[first_nonzero_elem_j, t] = multiplier

    P = A
    return L, P, U


'''
LPL' decomposition (Bruhat decomposition):

where
L is lower triangular,
L' is lower unitriangular,
P is transposition matrix.
'''


def lpl_decompose(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    # LecV.pdf p. 27
    Q = np.identity(A.shape[0])
    Q = Q[::-1]

    # AQ = LPU
    L, P, U = lpu_decompose(A @ Q)

    Ls = Q @ U @ Q
    return L, (P @ Q), Ls


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


def demo():
    test_matrices = [

        np.matrix([
            [7, 3, -1, 2],
            [3, 8, 1, -4],
            [-1, 1, 4, -1],
            [2, -4, -1, 6],
        ], dtype=float),

        np.matrix([
            [0, 1, 0],
            [-8, 8, 1],
            [2, -2, 0],
        ], dtype=float),

        np.matrix([
            [0, 1, 0],
            [-8, 8, 1],
            [2, -2, 0],
        ], dtype=float),

        # LUP / PLU, p.3: http://www.math.unm.edu/~loring/links/linear_s08/LU.pdf
        np.matrix([
            [2, 1, 0, 1],
            [2, 1, 2, 3],
            [0, 0, 1, 2],
            [-4, -1, 0, -2],
        ], dtype=float),

        # For Bruhat: http://mathpar.com/ru/help/08matrix.html
        np.matrix([
            [1, 4, 0, 1],
            [4, 5, 5, 3],
            [1, 2, 2, 2],
            [3, 0, 0, 1],
        ], dtype=float),

        np.matrix([
            [0, 2],
            [1, 4],
        ], dtype=float)

    ]

    A = test_matrices[1]

    print(A)
    print()

    L, P, U = lpu_decompose(A.copy())
    print("L")
    print(L)
    print("P")
    print(P)
    print("U")
    print(U)
    print("LPU")
    print(L @ P @ U)

    print()

    L, P, Ls = lpl_decompose(A.copy())
    print("L")
    print(L)
    print("P")
    print(P)
    print("L'")
    print(Ls)
    print("LPL")
    print(L @ P @ Ls)


if __name__ == "__main__":
    demo()
