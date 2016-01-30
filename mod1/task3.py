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
from task1 import PLU_decomposition

test_matrix = np.matrix([
    [1, 1, 1],
    [2, 2, 5],
    [4, 6, 8],
], dtype=np.float)

test_vector = np.array([
    1,
    2,
    3,
])

# First, we get PLU decomposition of A, such that PA=LU, so LUx=Pb
# Second, we solve the equation Ly=Pb for y by forward substitution
# Third, we solve the equation Ux=y for x
# https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
def plu_solve(A: np.matrix, b: np.array) -> np.array:
    A = A.astype(np.float)
    b = b.astype(np.float)

    P,L,U = PLU_decomposition(A)
    Pb = P.transpose().dot(b)

    y = np.zeros(b.size)
    for m in range(0, b.size):
        y[m] = Pb[m] - sum(
            L[m][i] * y[i] for i in range(0, m)
        )
        y[m] /= L[m][m]

    x = np.zeros(b.size)
    for m in reversed(range(0, b.size)):
        x[m] = y[m] - sum(
            U[m][i] * x[i] for i in range(m+1, b.size)
        )
        x[m] /= U[m][m]

    return x

if __name__ == '__main__':
    A = test_matrix
    b = test_vector

    print(plu_solve(A,b))
    print(npsolve(A, b))

