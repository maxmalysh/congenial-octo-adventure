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

#
# Про нормы, числа обусловленности и оценку ошибок:
#  http://www.math.hawaii.edu/~jb/math411/nation1
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve as npsolve
from task1 import PLU_decomposition, PLR_decomposition
from utils import test_matrices, MatrixBuilder, get_random_vector

test_matrix = np.matrix([
    [1, 1, 1],
    [2, 2, 5],
    [4, 6, 8],
], dtype=np.float)

test_matrix = test_matrices[0]
# test_matrix = np.matrix([
#     [50, 2, 900],
#     [0.1, 0.1, 14],
#     [1.0, 99, 1.99],
# ], dtype=np.float)

# test_matrix = np.matrix([
#     [10000, 0, 90000],
#     [0.0001, 0.1, 14],
#     [1.0, 99999, 1.999],
# ], dtype=np.float)

test_vector = np.array([
    1, 0, 0, 0
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
norm1 = lambda A: np.linalg.norm(A, ord=1)


def condition_number_for(A):
    return norm(A) * norm(np.linalg.inv(A))

# First, we get PLU decomposition of A, such that PA=LU, so LUx=Pb
# Second, we solve the equation Ly=Pb for y by forward substitution
# Third, we solve the equation Ux=y for x
# https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
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


# Оценщик числа обусловленности - condition number estimator
def cn_estimator(A: np.matrix):
    P, L, R = PLU_decomposition(A)

    n = A.shape[0]

    p = np.zeros(n, dtype=np.float)
    y = np.zeros(n, dtype=np.float)
    T = R.transpose()

    for k in reversed(range(0, n)):
        Tk = np.array([ T[k, i] for i in range(0, k) ], dtype=np.float)

        ykp = (1.0 - p[k]) / T[k, k]
        ykm = (-1.0 - p[k]) / T[k, k]

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
    # A.transpose().dot(P).dot(u), U.transpose().dot(L.transpose()).dot(u), y

    # Step 3:
    k_num = norm(A) * norm(z) / norm(r)
    return k_num


if __name__ == '__main__':
    A = test_matrix
    b = test_vector

    # print(npsolve(A, b))
    # print(plu_solve(A, b))

    print(cn_estimator(A))
    print(condition_number_for(A))

    nmatrices = 25
    msize = 30
    systems = []
    for i in range(0, nmatrices):
        matrix = MatrixBuilder(msize).nonsingular().nearsingular(3000).gen()
        vector = get_random_vector(msize)
        systems.append( (matrix, vector) )

    from task2 import relative_residual
    reals = [] ; estims = []; residues = []

    lvalues = []
    rvalues = []
    for i in range(0, len(systems)):
        matrix, vector = systems[i]

        x = npsolve(matrix.astype(np.float128), vector.astype(np.float128))
        xhat = plu_solve(matrix, vector)

        res = relative_residual(matrix, xhat, vector)

        real_cn = condition_number_for(matrix)
        estim_cn = cn_estimator(matrix)

        reals.append(real_cn)
        estims.append(estim_cn)
        residues.append(res * estim_cn)

        r = vector - matrix.dot(x)
        lvalues.append( norm(x - xhat) / norm(x) )
        rvalues.append( estim_cn * (norm(r) / norm(b)) )

    for i in range(0, len(systems)):
        print(reals[i], estims[i])
        #print(lvalues[i], rvalues[i]

    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.set_yscale('log')

    ax1.plot(reals)
    ax1.plot(estims)
    # ax2.plot(residues, color='r')
    # plt.plot(lvalues)
    # plt.plot(rvalues)

    plt.show()
