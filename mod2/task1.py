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


# Golub, van Loun 4rd ed., p. 632
def conj_gradients_method(A: np.matrix, b: np.ndarray, x_0: np.ndarray):
    x = {};
    r = {};
    beta = {};
    alpha = {};
    q = {};
    c = {};
    d = {};
    nu = {};
    l = {};

    x[0] = x_0
    r[0] = b - A @ x_0
    beta[0] = np.linalg.norm(r[0], ord=2)
    q[0] = 0
    c[0] = 0

    k = 0
    while (not math.isclose(beta[k], 0)) and (k < A.shape[0]):
        q[k + 1] = r[k] / beta[k]
        k += 1
        alpha[k] = q[k].transpose() @ A @ q[k]

        if k == 1:
            print(alpha[k])
            d[1] = alpha[1]
            nu[1] = beta[0] / d[1]
            c[k] = q[1]
        else:
            l[k - 1] = beta[k - 1] / d[k - 1]
            d[k] = alpha[k] - beta[k - 1] * l[k - 1]
            nu[k] = -beta[k - 1] * nu[k - 1] / d[k]
            c[k] = q[k] - l[k - 1] * c[k - 1]

        x[k] = x[k - 1] + nu[k] * c[k]
        r[k] = A @ q[k] - alpha[k] * q[k] - beta[k - 1] * q[k - 1]
        beta[k] = np.linalg.norm(r[k], ord=2)

    x_star = x[k]
    return x_star


def matrix_portrait(A: np.matrix) -> Set[Tuple[int, int]]:
    Omega = set()
    n = A.shape[0]
    for i in range(0, n):
        for j in range(0, n):
            if not math.isclose(A[i, j], 0):
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


def ilu_k(A: np.matrix, k: int, modified: bool) -> Tuple[np.matrix, np.matrix]:
    Omega = matrix_portrait(A)
    for i in range(0, k+1):
        L, R = incomplete_lu(A, Omega, modified)
        T = L @ R - A
        Omega |= matrix_portrait(T) # | is set union

    return L, R


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    A = np.array(
       [[ 6.,  0.,  0., -1., -1.],
       [ 0.,  8., -4., -2.,  0.],
       [ 0., -4.,  9.,  0.,  0.],
       [-1., -2.,  0.,  9.,  0.],
       [-1.,  0.,  0.,  0.,  7.]], dtype=float)

    L, R = ilu_k(A, 0, True)
    T = L@R - A
    L2, R2 = ilu_k(A, 0, False)
    T2 = L2@R2 - A
    print(L - L2)
    print(R - R2)
    print(T - T2)

    #print(L@R - A)

    #A = np.array([[3.0, -1.0],
    #              [-1.0, 3.0]])
    #b = np.array([1.0, 2.0])
    #x_0 = np.array([1000, 1000])
    #x_star = conj_gradients_method(A, b, x_0)
    #print(x_star)
