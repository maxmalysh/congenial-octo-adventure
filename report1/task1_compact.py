import numpy as np
from typing import List
from enum import Enum
from scipy import sparse
from random import randint


class PivotMode(Enum):
    BY_ROW = 0
    BY_COLUMN = 1
    BY_MATRIX = 2
    SPARSE = 3


def pivot_by_row(A: np.matrix, k: int, rp: List[float]):
    mat_size = A.shape[0]
    r_max_lead = k

    for i in range(k + 1, mat_size):
        if abs(A[rp[i], k]) > abs(A[rp[r_max_lead], k]):
            r_max_lead = i

    if r_max_lead != k:
        row_perm_k_saved = rp[k]
        rp[k] = rp[r_max_lead]
        rp[r_max_lead] = row_perm_k_saved


def pivot_by_column(A: np.matrix, k: int, cp: List[float]):
    mat_size = A.shape[0]
    c_max_lead = k

    for i in range(k + 1, mat_size):
        if abs(A[k, cp[i]]) > abs(A[k, cp[c_max_lead]]):
            c_max_lead = i

    if c_max_lead != k:
        column_perm_k_saved = cp[k]
        cp[k] = cp[c_max_lead]
        cp[c_max_lead] = column_perm_k_saved


def pivot_by_matrix(A: np.matrix, k: int, rp: List[float], cp: List[float]):
    mat_size = A.shape[0]
    r_max_lead = k
    c_max_lead = k

    for i in range(k + 1, mat_size):
        for j in range(k + 1, mat_size):
            if abs(A[rp[i], cp[j]]) > abs(
                    A[rp[r_max_lead], cp[c_max_lead]]):
                r_max_lead = i
                c_max_lead = j

    if r_max_lead != k:
        row_perm_k_saved = rp[k]
        rp[k] = rp[r_max_lead]
        rp[r_max_lead] = row_perm_k_saved
    if c_max_lead != k:
        column_perm_k_saved = cp[k]
        cp[k] = cp[c_max_lead]
        cp[c_max_lead] = column_perm_k_saved


def reverse_permutation(perm: List[int]):
    rev_perm = [0] * len(perm)
    for i in range(len(perm)):
        rev_perm[perm[i]] = i
    return rev_perm


def pivot_sparse(A: sparse.dok_matrix, k: int, rp: List[float], cp: List[float]):
    n = A.shape[0]

    B = np.zeros((n - k, n - k),
                 dtype=np.integer)
    B_ = np.zeros((n - k, n - k),
                  dtype=np.integer)

    rp_rev = reverse_permutation(rp)
    cp_rev = reverse_permutation(cp)

    A_keys = A.keys() if isinstance(A, sparse.spmatrix) else [
        x for x in np.ndindex(A.shape) if A[x] != 0
        ]
    for i_permuted, j_permuted in A_keys:
        i = rp_rev[i_permuted] - k
        j = cp_rev[j_permuted] - k

        if i < 0 or j < 0:
            continue

        if A[i_permuted, j_permuted] != 0:
            B[i, j] = 1
        else:
            B_[i, j] = 1

    G = B @ B_.transpose() @ B

    epsilon = 1e-08

    g_min_idxs = []

    A_keys = A.keys() if isinstance(A, sparse.spmatrix) else [
        x for x in np.ndindex(A.shape) if A[x] != 0
        ]
    for i_permuted, j_permuted in A_keys:
        i = rp_rev[i_permuted] - k
        j = cp_rev[j_permuted] - k

        if i < 0 or j < 0:
            continue

        if abs(A[i_permuted, j_permuted]) < epsilon:
            continue
        if len(g_min_idxs) == 0:
            g_min_idxs.append((i, j))
        if G[i, j] < G[g_min_idxs[0]]:
            g_min_idxs.clear()
            g_min_idxs.append((i, j))
        elif G[i, j] == G[g_min_idxs[0]]:
            g_min_idxs.append((i, j))

    if len(g_min_idxs) == 0:
        raise ValueError("No non-almost-zero elements in A submatrix")

    g_chosen = g_min_idxs[0]
    for g_idx in g_min_idxs:
        if abs(A[rp[g_idx[0] + k], cp[g_idx[1] + k]]) \
                > abs(A[rp[g_chosen[0] + k], cp[g_chosen[1] + k]]):
            g_chosen = g_idx

    r_chosen_lead = g_chosen[0] + k
    c_chosen_lead = g_chosen[1] + k

    row_perm_k_saved = rp[k]
    rp[k] = rp[r_chosen_lead]
    rp[r_chosen_lead] = row_perm_k_saved

    column_perm_k_saved = cp[k]
    cp[k] = cp[c_chosen_lead]
    cp[c_chosen_lead] = column_perm_k_saved


def pivot(A: np.matrix, k: int, mode: PivotMode, rp: List[float], cp: List[float]):
    if mode == PivotMode.BY_ROW:
        pivot_by_row(A, k, rp)
    elif mode == PivotMode.BY_COLUMN:
        pivot_by_column(A, k, cp)
    elif mode == PivotMode.BY_MATRIX:
        pivot_by_matrix(A, k, rp, cp)
    elif mode == PivotMode.SPARSE:
        pivot_sparse(A, k, rp, cp)


def lu_decompose_pivoting(A: np.matrix, mode: PivotMode):
    n = A.shape[0]
    rp = [i for i in range(n)]
    cp = [i for i in range(n)]

    for k in range(0, n):
        pivot(A, k, mode, rp, cp)

        for j in range(k + 1, n):
            A[rp[k], cp[j]] /= A[rp[k], cp[k]]

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[rp[i], cp[j]] -= A[rp[k], cp[j]] * A[rp[i], cp[k]]

    return A, rp, cp


def lpu_decompose(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    A = A.copy()
    n = A.shape[0]
    L = np.identity(A.shape[0])
    U = np.identity(A.shape[0])

    for k in range(0, n):
        nonzero_elems_j_indices = np.nonzero(A[k])[0]

        if len(nonzero_elems_j_indices) != 0:
            first_nonzero_elem_j = nonzero_elems_j_indices[0]
        else:
            raise ValueError("Singular matrix provided!")

        first_nonzero_elem = A[k, first_nonzero_elem_j]

        for j in range(0, n):
            A[k, j] /= first_nonzero_elem

        L[k, k] = first_nonzero_elem

        for s in range(k + 1, n):
            multiplier = A[s, first_nonzero_elem_j]
            for j in range(0, n):
                A[s, j] -= A[k, j] * multiplier
            L[s, k] = multiplier

        for t in range(first_nonzero_elem_j + 1, n):
            multiplier = A[k, t] / A[k, first_nonzero_elem_j]
            for i in range(0, n):
                A[i, t] -= A[i, first_nonzero_elem_j] * multiplier

            U[first_nonzero_elem_j, t] = multiplier

    P = A
    return L, P, U


def lpl_decompose(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    Q = np.identity(A.shape[0])
    Q = Q[::-1]

    L, P, U = lpu_decompose(A @ Q)

    Ls = Q @ U @ Q
    return L, (P @ Q), Ls


def lu_extract(LU: np.matrix, rp: List[float], cp: List[float]):
    mat_size = LU.shape[0]
    L = np.zeros((mat_size, mat_size))
    U = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        for j in range(mat_size):
            if j > i:
                U[i, j] = LU[rp[i], cp[j]]
            elif j < i:
                L[i, j] = LU[rp[i], cp[j]]
            elif j == i:
                L[i, j] = LU[rp[i], cp[j]]
                U[i, j] = 1
    return L, U


def lr_decompose_pivoting(A: np.matrix, mode: PivotMode):
    n = A.shape[0]
    rp = [i for i in range(n)]
    cp = [i for i in range(n)]

    for k in range(0, n):
        pivot(A, k, mode, rp, cp)
        for i in range(k + 1, n):
            divider = A[rp[i], cp[k]] / A[rp[k], cp[k]]
            A[rp[i], cp[k]] = divider

            for j in range(k + 1, n):
                A[rp[i], cp[j]] -= A[rp[k], cp[j]] * divider

    return A, rp, cp


def lr_extract(LU: np.matrix, rp: List[float], cp: List[float]):
    mat_size = LU.shape[0]
    L = np.zeros((mat_size, mat_size))
    U = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        for j in range(mat_size):
            if j > i:
                U[i, j] = LU[rp[i], cp[j]]
            elif j < i:
                L[i, j] = LU[rp[i], cp[j]]
            elif j == i:
                U[i, j] = LU[rp[i], cp[j]]
                L[i, j] = 1
    return L, U


def perm_vector_to_matrix(vector, row=True):
    n = len(vector)
    matrix = np.zeros(shape=(n, n))

    for i in range(0, n):
        if row:
            matrix[vector[i], i] = 1
        else:
            matrix[i, vector[i]] = 1

    return matrix


def PLU_decomposition(A):
    mode = PivotMode.BY_ROW
    lu_in_one, P, P_ = lu_decompose_pivoting(A.copy(), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, U


def LUP_decomposition(A):
    mode = PivotMode.BY_COLUMN
    lu_in_one, P, P_ = lu_decompose_pivoting(A.copy(), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return L, U, perm_vector_to_matrix(P_, row=False)


def PLUP_decomposition(A):
    mode = PivotMode.BY_MATRIX
    lu_in_one, P, P_ = lu_decompose_pivoting(A.copy(), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, U, \
           perm_vector_to_matrix(P_, row=False)


def PLR_decomposition(A):
    mode = PivotMode.BY_ROW
    lr_in_one, P, P_ = lr_decompose_pivoting(A.copy(), mode)
    L, R = lr_extract(lr_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, R


def Sparse_decomposition(A):
    mode = PivotMode.SPARSE
    lu_in_one, P, P_ = lu_decompose_pivoting(A.copy(), mode)
    L, U = lu_extract(lu_in_one, P, P_)
    return perm_vector_to_matrix(P, row=True), L, U, \
           perm_vector_to_matrix(P_, row=False)
