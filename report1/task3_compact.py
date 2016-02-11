import numpy as np
from scipy.linalg import solve as npsolve
from task1 import PLU_decomposition, PLR_decomposition
from utils import test_matrices, MatrixBuilder

def max_row_norm(A):
    if len(A.shape) == 1:
        return max([abs(x) for x in A])

    m, n = A.shape[0], A.shape[1]

    absolute_sums = [
        sum([abs(A[i, j]) for j in range(0, n)]) for i in range(0, m)
    ]

    return max(absolute_sums)


# norm = max_row_norm
norm = lambda A: np.linalg.norm(A, ord=np.inf)
norm1 = lambda A: np.linalg.norm(A, ord=1)

def condition_number_for(A):
    return norm(A) * norm(np.linalg.inv(A))

def plu_solve(A: np.matrix, b: np.array) -> np.array:
    P, L, R = PLR_decomposition(A)
    Pb = P.transpose().dot(b)

    y = np.zeros(b.size)
    for m in range(0, b.size):
        y[m] = Pb[m] - sum(
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

def cn_estimator(A: np.matrix):
    P, L, R = PLR_decomposition(A)

    # Step 1:
    n = A.shape[0]

    p = np.zeros(n)
    y = np.zeros(n)
    T = R.transpose()

    for k in reversed(range(0, n)):
        Tk = np.array([ T[k, i] for i in range(0, k) ])

        ykp = (1 - p[k]) / T[k, k]
        ykm = (-1 - p[k]) / T[k, k]

        pkp = p[:k] + Tk*ykp
        pkm = p[:k] + Tk*ykm

        if abs(ykp) + norm1(pkp) >= abs(ykm) + norm1(pkm):
            y[k] = ykp
            p[:k] = pkp

        else:
            y[k] = ykm
            p[:k] = pkm

    # Step 2:
    r = npsolve(L.transpose(), y)
    w = npsolve(L, P.transpose().dot(r))
    z = npsolve(R, w)

    # Step 3:
    k_num = norm(A) * norm(z) / norm(r)
    return k_num


