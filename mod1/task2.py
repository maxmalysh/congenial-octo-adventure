'''
# 1) Реализовать метод Гаусса с оптимальным заполнением для разреженных матриц и
#    построением соответсвующего этому методу разложения матрицы системы A вида A=PLUP'; - DONE
# 2) при реализации учесь формат хранения A;
# 3) реализовать для этого метода итерационное уточнение простой и переменной точности,
#    оценить качество получаемых результатов;
'''



import numpy as np
from scipy.linalg import solve as npsolve
from task1 import PLUP_decomposition

test_matrix = np.matrix([
    [1, 1, 1],
    [2, 2, 5],
    [20, 6, 8],
], dtype=np.float)

test_vector = np.array([
    1,
    2,
    3,
])


def plup_solve(A: np.matrix, b: np.array) -> np.array:
    A = A.astype(np.float)
    b = b.astype(np.float)

    P,L,U,P_ = PLUP_decomposition(A)
    Pb = P.transpose().dot(b)

    y = np.zeros(b.size)
    for m in range(0, b.size):
        y[m] = Pb[m] - sum(
            L[m][i] * y[i] for i in range(0, m)
        )
        y[m] /= L[m][m]

    z = np.zeros(b.size)
    for m in reversed(range(0, b.size)):
        z[m] = y[m] - sum(
            U[m][i] * z[i] for i in range(m+1, b.size)
        )
        z[m] /= U[m][m]

    x = P_.transpose().dot(z)

    return x

if __name__ == '__main__':
    A = test_matrix
    b = test_vector

    print(plup_solve(A,b))
    print(npsolve(A, b))
    print(
        np.allclose(plup_solve(A,b), npsolve(A, b))
    )