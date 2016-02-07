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
        row_perm_k_saved = row_permutation[k]
        row_permutation[k] = row_permutation[row_with_max_leading_elem]
        row_permutation[row_with_max_leading_elem] = row_perm_k_saved


def pivot_by_column(A: np.matrix, k: int, column_permutation: List[float]):
    mat_size = A.shape[0]
    column_with_max_leading_elem = k

    for i in range(k + 1, mat_size):
        if abs(A[k, column_permutation[i]]) > abs(A[k, column_permutation[column_with_max_leading_elem]]):
            column_with_max_leading_elem = i

    if column_with_max_leading_elem != k:
        column_perm_k_saved = column_permutation[k]
        column_permutation[k] = column_permutation[column_with_max_leading_elem]
        column_permutation[column_with_max_leading_elem] = column_perm_k_saved


def pivot_by_matrix(A: np.matrix, k: int, row_permutation: List[float], column_permutation: List[float]):
    mat_size = A.shape[0]
    row_with_max_leading_elem = k
    column_with_max_leading_elem = k

    for i in range(k + 1, mat_size):
        for j in range(k + 1, mat_size):
            if abs(A[row_permutation[i], column_permutation[j]]) > abs(
                    A[row_permutation[row_with_max_leading_elem], column_permutation[column_with_max_leading_elem]]):
                row_with_max_leading_elem = i
                column_with_max_leading_elem = j

    if row_with_max_leading_elem != k:
        row_perm_k_saved = row_permutation[k]
        row_permutation[k] = row_permutation[row_with_max_leading_elem]
        row_permutation[row_with_max_leading_elem] = row_perm_k_saved
    if column_with_max_leading_elem != k:
        column_perm_k_saved = column_permutation[k]
        column_permutation[k] = column_permutation[column_with_max_leading_elem]
        column_permutation[column_with_max_leading_elem] = column_perm_k_saved


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
LPU decomposition (modified Bruhat decomposition) for nonsingular matrix:

where
L is lower triangular,
U is upper unitriangular,
P is transposition matrix.
(like in the task).

# Not an in-place algorithm.
'''


def lpu_decompose(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    A = A.astype(np.float)
    n = A.shape[0]
    L = np.identity(A.shape[0])
    U = np.identity(A.shape[0])

    # The algorithm: LecV.pdf p. 27.
    for k in range(0, n):  # k - Gaussian step number
        nonzero_elems_j_indices = np.nonzero(A[k])[0]  # Returns a tuple of arrays, one for each dimension of a,
        # containing the indices of the non-zero elements in that dimension. Get indices in dimension 0.

        if len(nonzero_elems_j_indices) != 0:
            first_nonzero_elem_j = nonzero_elems_j_indices[0]  # Get first index.
        else:
            raise ValueError("Singular matrix provided. LPU only works for nonsingular matrices.")

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
LPL' decomposition (Bruhat decomposition) for nonsingular matrix:

where
L is lower triangular,
L' is lower unitriangular,
P is transposition matrix.
'''


def lpl_decompose(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    A = A.astype(np.float)
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

def lr_decompose_pivoting(A: np.matrix, mode: PivotMode):
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


def lr_extract(LU: np.matrix, row_permutation: List[float], column_permutation: List[float]):
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
    from tests import test_matrices
    A = test_matrices[4].astype(np.float)

    print(A)
    print()

    lu_in_one, P, P_ = lu_decompose_pivoting(A.copy(), PivotMode.BY_MATRIX)
    L, U = lu_extract(lu_in_one, P, P_)
    print(P)
    print(P_)
    print(L @ U)


if __name__ == "__main__":
    demo()


'''
@vector - permutation vector as a list, e.g. [2,0,1]
@row – boolean determining whether vector is shuffling rows or columns
'''
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

def PLR_decomposition(A):
    mode = PivotMode.BY_ROW
    lr_in_one, P, P_ = lr_decompose_pivoting(A.astype(np.float), mode)
    L, R = lr_extract(lr_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, R

