'''
1) Реализовать метод Гаусса с выбором по столбцу - Done
2) построить с его помощью оценщик числа обусловленности матрицы системы в строчной норме;
3) протестировать качество данного оценщика и получаемой с его помощью
   оценкой относительной погрешности решения через вектор невязки.
стр.36 (снизу) - оценщик срочной матричной нормы
'''

# стр. 32:
# Решение линейной системы Ax=b с использованием метода Гаусса с выбором по столбцу
# сводится к решению двух треугольных систем:
# | Ly = Pb
# | Ux = y
# где PA = LU

import numpy as np
from scipy.linalg import solve as npsolve
from task1 import PLU_decomposition, PLR_decomposition

test_matrix = np.matrix([
    [1, 1, 1],
    [2, 2, 5],
    [4, 6, 8],
], dtype=np.float)

test_matrix = np.matrix([
    [50, 2, 900],
    [0.1, 0.1, 14],
    [1.0, 99, 1.99],
], dtype=np.float)

# test_matrix = np.matrix([
#     [10000, 0, 90000],
#     [0.0001, 0.1, 14],
#     [1.0, 99999, 1.999],
# ], dtype=np.float)

test_vector = np.array([
    1, 0, 0,
    # 1, 2, 3,
])


def max_row_norm(A):
    if len(A.shape) == 1:
        return max([abs(x) for x in A])

    m, n = A.shape[0], A.shape[1]  # m строк, n столбцов

    absolute_sums = [
        sum([abs(A[i, j]) for j in range(0, n)]) for i in range(0, m)
        ]

    return max(absolute_sums)


# norm = max_row_norm
norm = lambda A: np.linalg.norm(A, ord=np.inf)


def condition_number_for(A):
    return norm(A) * norm(np.linalg.inv(A))

# First, we get PLU decomposition of A, such that PA=LU, so LUx=Pb
# Second, we solve the equation Ly=Pb for y by forward substitution
# Third, we solve the equation Ux=y for x
# https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
def plu_solve(A: np.matrix, b: np.array) -> np.array:
    A = A.astype(np.float)
    b = b.astype(np.float)

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

    # Solve
    # A^{t}*x = e
    # Ay =  x
    # Here vector e is chosen during the first step of the solution,
    # finding z such that U^{t} * z = e
    # Each element e_{i} is +/- 1


    # return x

    # Step 1:
    xhat, yhat = x, y

    # Step 2:
    u = npsolve(L.transpose(), xhat)
    w = npsolve(L, P.transpose().dot(u))
    z = npsolve(R, w)

    # A.transpose().dot(P).dot(u), U.transpose().dot(L.transpose()).dot(u), y
    k_num = norm(A) * norm(z) / norm(u)
    return x, k_num


#
# Оценщик числа обусловленности - condition number estimator
#
def cn_estimator(A: np.matrix, b: np.array):
    A = A.astype(np.float)
    b = b.astype(np.float)

    P, L, R = PLR_decomposition(A)
    Pb = P.transpose().dot(b)

    # Forward substitution
    y = np.zeros(b.size)
    for m in range(0, b.size):
        y[m] = Pb[m] - sum(
                L[m][i] * y[i] for i in range(0, m)
        )
        y[m] /= L[m][m]

    # Backward substitution (step 1)
    x = np.zeros(b.size)
    p = np.zeros(b.size)
    yhat = np.zeros(b.size)
    xhat = np.zeros(b.size)
    p = np.zeros(b.size)

    for k in reversed(range(0, b.size)):
        x[k] = y[k] - sum(
                R[k][i] * x[i] for i in range(k + 1, b.size)
        )
        x[k] /= R[k][k]

        p[k] = sum(R[k][i] * x[i] for i in range(k + 1, b.size))
        #        p[0:k] = pl(k) + x[k]*tk(k)

        tk = lambda n: np.array([R[k][i] for i in range(0, n + 1)])
        pl = lambda n: np.array([p[i] for i in range(0, n + 1)])

        yhat[k] = 0.0  # ...

        xkp = (1 - p[k]) / R[k][k]
        xkm = (-1 - p[k]) / R[k][k]

        skp = abs(xkp) + norm(pl(k) + xkp * tk(k))
        skm = abs(xkm) + norm(pl(k) + xkm * tk(k))

        xhat[k] = xkp if skp >= skm else xkm

    # norm(xhat) should be close to norm(R.I)

    # Step 2:
    u = npsolve(L.transpose(), xhat)
    w = npsolve(L, P.transpose().dot(u))
    z = npsolve(R, w)

    # A.transpose().dot(P).dot(u), U.transpose().dot(L.transpose()).dot(u), y
    k_num = norm(A) * norm(z) / norm(u)
    return x, k_num


if __name__ == '__main__':
    A = test_matrix
    b = test_vector

    print(npsolve(A, b))
    print(plu_solve(A, b))
    print(cn_estimator(A,b))

    print(condition_number_for(A))
    print(np.linalg.cond(A, np.inf))
