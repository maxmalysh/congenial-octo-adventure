# 1.
# Реализовать предобусловленный метод сопряженных градиентов для систем с матрицами Стилтьеса
# с предобуславливанием по методам ILU(k), MILU(k) и ILU(k,e)
#
# (в последнем случае речь идёт об алгоритме ILU(k), в котором портрет матрицы заменён на множетсов пар индексов,
#  включающее пары равных индексов и пары индексов коэффициентов матрицы по модулю больших e);
#
# провести анализ скорости сходимости для заданной системы и подобрать приемлемые значения k.
#

#
# стр. 90 – Метод сопряжённых градиентов
# стр. 102 - Предобусловленный метод сопряженных градиентов
# стр. 112 – Предобуславливание с использованием неполного LU-разложения
# стр. 113 – определение матриц Стилтьеса
# стр. 119 – определение ILU(k) разложения
# стр. 120 – MILU-разложение
#
# Предобуславливание (также предобусловливание) — процесс преобразования условий задачи
# для её более корректного численного решения.
#
# Предобуславливание обычно связано с уменьшением числа обусловленности задачи.
# Предобуславливаемая задача обычно затем решается итерационным методом.
#
# https://ru.wikipedia.org/wiki/Метод_сопряжённых_градиентов
# Метод сопряженных градиентов — метод нахождения локального экстреммума функции
# на основе информации о её значениях и её градиенте.
#
#
#
#
import numpy as np
import math
from typing import Tuple, Set
import matplotlib.pyplot as plt
import pickle


from mod1.utils import MatrixBuilder, get_random_vector

filename = "/Users/o2genum/Downloads/systems2.bin"

def get_systems():
    dense_number = 0
    sparse_number = 200

    # Try to read some already generated matrices
    try:
        with open(filename, 'rb') as input:
            system_set_dense, system_set_sparse = pickle.load(input)
    except FileNotFoundError:
        system_set_dense, system_set_sparse = [], []

    if len(system_set_dense) == dense_number and len(system_set_sparse) == sparse_number:
        print("Got %d dense and %d sparse systems" % (len(system_set_dense), len(system_set_sparse)))
        return system_set_dense, system_set_sparse

    # Otherwise generate new matrices
    for i in range(0, dense_number):
        print('.', end='', flush=True)
        matrix = MatrixBuilder(4).stieltjes().nonsingular().gen()
        vector = get_random_vector(4)
        system_set_dense.append( (matrix, vector) )

    for i in range(0, sparse_number):
        print('x', end='', flush=True)
        matrix = MatrixBuilder(9).sparse_stieltjes().nonsingular().gen()
        vector = get_random_vector(9)
        system_set_sparse.append( (matrix, vector) )
    print('\n')

    # And save them for the next time
    with open(filename, 'wb') as output:
        pickle.dump([system_set_dense, system_set_sparse], output, pickle.HIGHEST_PROTOCOL)

    return system_set_dense, system_set_sparse

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

        if k > 300:
            raise ValueError("Does not converge")

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

        if k > 300:
            raise ValueError("Does not converge")

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


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    dense_set, sparse_set = get_systems()
    test_set = sparse_set

    conv_mat_count = 0
    for mat_i in range(0, len(test_set)):

        try:
            A = test_set[mat_i][0]
            b = test_set[mat_i][1]

            x_0 = np.array([0] * A.shape[0], dtype=float)

            x_star, k = conj_grad(A.copy(), b, x_0)
            #print("CONJ GRAD, iterations: " + str(k))

            iter_n = 10

            plainplot = [k] * iter_n

            iluplot = []
            miluplot = []

            for k in range(0, iter_n):
                #x_star, iters = conj_grad_precond(A.copy(), b, x_0, lambda A: ilu_k(A, 0, e=0.00000001*(10**k)))
                #print("PRECOND CONJ GRAD, iterations: " + str(iters))
                #iluplot.append(iters)
                pass
            plt.plot(plainplot, color='g')
            #plt.plot(iluplot, color='b')

            conv_mat_count += 1
        except ValueError as e:
            continue
    print("Conv matrices: " + str(conv_mat_count))
    plt.show()
