import numpy as np
import math
from typing import Tuple, Set

def conj_grad_precond(A: np.matrix, b: np.ndarray, x_0: np.ndarray, precond_func):
    k = 0
    r = {}; r[0] = A @ x_0 - b
    x = {}; x[0] = x_0
    z = {}
    p = {}

    L, U = precond_func(A)
    M = L @ U
    z[0] = np.linalg.solve(M, r[0])

    while not math.isclose(np.linalg.norm(r[k], ord=2), 0):
        k += 1
        if k == 1:
            p[k] = z[0]
        else:
            tau = (r[k-1].transpose() @ z[k-1]) / (r[k-2].transpose() @ z[k-2])
            p[k] = z[k-1] + tau * p[k-1]
        mu = (r[k-1].transpose() @ z[k-1]) / (p[k].transpose() @ A @ p[k])
        x[k] = x[k-1] - mu * p[k]
        r[k] = r[k-1] - mu * (A @ p[k])
        z[k] = np.linalg.solve(M, r[k])

    x_star = x[k]
    return x_star, k

def matrix_portrait(A: np.matrix, e: float = None) -> Set[Tuple[int, int]]:
    if e is None:
        Omega = set()
        n = A.shape[0]
        for i in range(0, n):
            for j in range(0, n):
                if not math.isclose(A[i, j], 0):
                    Omega.add((i, j))

        return Omega
    else:
        Omega = set()
        n = A.shape[0]
        for i in range(0, n):
            for j in range(0, n):
                if abs(A[i, j]) > e or i==j:
                    Omega.add((i, j))

        return Omega

def s_compute(R: np.matrix, i: int):
    n = R.shape[0]
    s_i = 0
    for j in range(0, n):
        if j != i:
            s_i += R[i, j]
    return s_i

def incomplete_lu(A: np.matrix, Omega: Set[Tuple[int, int]], modified: bool) -> Tuple[np.matrix, np.matrix]:
    A = A.copy()
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    R = np.zeros(A.shape, dtype=float)

    for k in range(0, n):
        for i in range(k, n):
            R[k, i] = A[k, i] if (k, i) in Omega else 0
        for j in range(k + 1, n):
            L[j, k] = A[j, k] / A[k, k] if (j, k) in Omega else 0
        for p in range(k + 1, n):
            for q in range(k + 1, n):
                if not modified:
                    A[p, q] -= L[p, k]*R[k, q]
                else:
                    A[p, q] -= L[p, k]*(R[k, q] - (s_compute(R, k) if p==q else 0))

    return L, R


def ilu_k(A: np.matrix, k: int, modified: bool = False, e: float = None) -> Tuple[np.matrix, np.matrix]:
    Omega = matrix_portrait(A, e)
    for i in range(0, k+1):
        L, R = incomplete_lu(A, Omega, modified)
        T = L @ R - A
        Omega |= matrix_portrait(T, e) # | is set union

    return L, R

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    A = np.array(
       [[ 77., -23., -32.,   0.,   0.],
       [-23.,  53.,   0.,   0., -18.],
       [-32.,   0.,  90.,  -5.,   0.],
       [  0.,   0.,  -5.,  49., -15.],
       [  0., -18.,   0., -15.,  89.]], dtype=float)

    b = np.array([1, 1, 1, 1, 1], dtype=float)
    x_0 = np.array([0, 0, 0, 0, 0], dtype=float)

    x_star = np.linalg.solve(A.copy(), b)
    print("NUMPY:")
    print(x_star)

    x_star, k = conj_grad_precond(A.copy(), b, x_0, lambda A: ilu_k(A, 10))
    print("PRECOND CONJ GRAD, iterations: " + str(k))
    print(x_star)
