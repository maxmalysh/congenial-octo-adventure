import numpy as np
import math
from typing import Tuple, Set


def conj_grad(A: np.matrix, b: np.ndarray, x_0: np.ndarray):
    k = 0
    r = {}; r[0] = b - A @ x_0
    x = {}; x[0] = x_0
    p = {}
    tau = {}
    mu = {}

    while not math.isclose(np.linalg.norm(r[k], ord=2), 0):
        k += 1
        if k == 1:
            p[k] = r[0]
        else:
            tau[k-1] = (r[k-1].transpose() @ r[k-1]) / (r[k-2].transpose() @ r[k-2])
            p[k] = r[k-1] + tau[k-1] * p[k-1]
        mu[k] = (r[k-1].transpose() @ r[k-1]) / (p[k].transpose() @ A @ p[k])
        x[k] = x[k-1] + mu[k] * p[k]
        r[k] = r[k-1] - mu[k] * (A @ p[k])

    x_star = x[k]
    return x_star, k


def lu_solve(L: np.matrix, R: np.matrix, b: np.array) -> np.array:
    y = np.zeros(b.size)
    for m in range(0, b.size):
        y[m] = b[m] - sum(
            L[m][i] * y[i] for i in range(0, m)
        )
        y[m] /= L[m][m]

    x = np.zeros(b.size)
    for k in reversed(range(0, b.size)):
        x[k] = y[k] - sum(
                R[k][i] * x[i] for i in range(k + 1, b.size)
        )
        x[k] /= R[k][k]

    return x


def conj_grad_precond(A: np.matrix, b: np.ndarray, x_0: np.ndarray, precond_func):
    k = 0
    r = {}; r[0] = b - A @ x_0
    x = {}; x[0] = x_0
    z = {}
    p = {}

    L, U = precond_func(A)
    z[0] = lu_solve(L, U, r[0])

    while not math.isclose(np.linalg.norm(r[k], ord=2), 0):
        k += 1
        if k == 1:
            p[k] = z[0]
        else:
            tau = (r[k-1].transpose() @ z[k-1]) / (r[k-2].transpose() @ z[k-2])
            p[k] = z[k-1] + tau * p[k-1]
        mu = (r[k-1].transpose() @ z[k-1]) / (p[k].transpose() @ A @ p[k])
        x[k] = x[k-1] + mu * p[k]
        r[k] = r[k-1] - mu * (A @ p[k])
        z[k] = lu_solve(L, U, r[k])

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


def incomplete_lu(A: np.matrix, Omega: Set[Tuple[int, int]], modified: bool = False) -> Tuple[np.matrix, np.matrix]:
    A = A.copy()
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    R = np.zeros(A.shape, dtype=float)

    for k in range(0, n):
        for i in range(k, n):
            if (k, i) in Omega:
                R[k, i] = A[k, i]
            elif modified:
                R[k, k] -= A[k, i]
                R[k, i] = 0
        for j in range(k + 1, n):
            L[j, k] = A[j, k] / A[k, k] if (j, k) in Omega else 0
        for p in range(k + 1, n):
            for q in range(k + 1, n):
                    A[p, q] -= L[p, k]*R[k, q]

    return L, R


def ilu_k(A: np.matrix, k: int, modified: bool = False, e: float = None) -> Tuple[np.matrix, np.matrix]:
    Omega = matrix_portrait(A, e)
    for i in range(0, k+1):
        L, R = incomplete_lu(A, Omega, modified)
        T = L @ R - A
        Omega |= matrix_portrait(T, e) # | is set union

    return L, R