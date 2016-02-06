'''
# 1) Реализовать метод Гаусса с оптимальным заполнением для разреженных матриц и
#    построением соответсвующего этому методу разложения матрицы системы A вида A=PLUP'; - DONE
# 2) при реализации учесь формат хранения A;
# 3) реализовать для этого метода итерационное уточнение простой и переменной точности,
#    оценить качество получаемых результатов; - DONE
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
    A = A.astype(np.float32)
    b = b.astype(np.float32)

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


# Итерационное уточнение - стр. 34, два пункта сверху
# Iterative refinement
def plup_solve_iterative(A: np.matrix, b: np.array, double_precision=False) -> np.array:
    x = plup_solve(A, b)

    iteration_count = 2

    for i in range(0, iteration_count):
        chosen_type = np.longfloat if double_precision else np.float
        r = b.astype(chosen_type) - A.astype(chosen_type).dot(x).astype(chosen_type)
        z = plup_solve(A, r)
        x = x + z

    return x

# Average solution deviation (absolute) for system Ax=b
def solution_deviation(A:np.matrix, x:np.array, b:np.array):
    differences = b - A.dot(x)
    return np.mean( [abs(err) for err in differences] )


if __name__ == '__main__':
    #
    # Checking whether solver works
    #

    A = test_matrix
    b = test_vector

    print(plup_solve(A,b))
    print(npsolve(A, b))
    print(
        np.allclose(plup_solve(A,b), npsolve(A, b))
    )

    #
    # Checking whether iterative solver works better
    #
    msize = 100

    singular = True

    while singular:
        matrix = np.random.randint(low=-100, high=100, size=(msize, msize))
        if np.linalg.det(matrix) != 0:
            singular = False

    vector = np.random.randint(low=-100, high=100, size=msize)

    numpyResult = npsolve(matrix.astype(np.float64), vector.astype(np.float64))
    ourResult = plup_solve(matrix, vector)
    ourBetterResult = plup_solve_iterative(matrix, vector, double_precision=False)
    ourEvenBetterResult = plup_solve_iterative(matrix, vector, double_precision=True)

    results = [numpyResult, ourResult, ourBetterResult, ourEvenBetterResult]
    for result in results:
        print(solution_deviation(matrix, result, vector))

