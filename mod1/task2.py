'''
# 1) Реализовать метод Гаусса с оптимальным заполнением для разреженных матриц и
#    построением соответсвующего этому методу разложения матрицы системы A вида A=PLUP'; - DONE
# 2) при реализации учесь формат хранения A; - DONE
# 3) реализовать для этого метода итерационное уточнение простой и переменной точности,
#    оценить качество получаемых результатов; - DONE
'''

import numpy as np
from scipy.linalg import solve as npsolve
from scipy.sparse.base import spmatrix
import utils
from utils import MatrixBuilder
from task1 import PLUP_decomposition, Sparse_decomposition

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

#
# Solve equation system Ax=b by using A=PLUP' decomposition
#
# Если аргумент sparse не указан, то выбор разложения выбирается автоматически
# по типу матрицы A.
#
# Если хочется заставить функцию использовать
# PLUP разложение для sparse матрицы (или sparse разложение для np.matrix),
# то можно указать plup_solve(..., ..., sparse=True/False)
#
def plup_solve(A: np.matrix, b: np.array, sparse=None) -> np.array:
    A = A
    b = b

    if sparse == None:
        sparse = True if isinstance(A, spmatrix) else False

    P,L,U,P_ = Sparse_decomposition(A) if sparse else PLUP_decomposition(A)
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
def iterative_refinement(A: np.matrix, b: np.array, solver, iterations=1, double_precision=False, sparse=False) -> np.array:
    x = solver(A, b) if not sparse else solver(A, b, sparse=True)

    for i in range(0, iterations):
        chosen_type = np.longfloat if double_precision else A.dtype
        r = b.astype(chosen_type) - A.astype(chosen_type).dot(x).astype(chosen_type)
        z = solver(A, r.astype(A.dtype))
        x = x + z

    return x

#
# Average solution deviation (absolute) for system Ax=b
# Среднее значение модулей элементов вектора невязки
#
def solution_deviation(A:np.matrix, x:np.array, b:np.array):
    differences = b - A.dot(x)
    return np.mean( [abs(err) for err in differences] )


#
# Generates non-singular equation system in form Ax=b,
#
def generate_test_system(size=100, sparse=False):
    vector = utils.get_random_vector(size, low=-100000, high=100000)

    if not sparse:
        matrix = MatrixBuilder(size, low=-100000, high=100000).nonsingular().gen()
    else:
        matrix = MatrixBuilder(size).nonsingular().randsparse().gen()

    return matrix, vector

def check_whether_custum_solver_works(solver):
    A, b = test_matrix, test_vector
    return np.allclose(plup_solve(A,b), npsolve(A, b))

if __name__ == '__main__':
    # Here we check how precise are solutions obtained by different solvers
    A, b = generate_test_system(size=100, sparse=True)

    numpyResult = npsolve(A, b)
    ourResult = plup_solve(A, b, sparse=False)
    sparseResult = plup_solve(A, b, sparse=True)

    ourBetterResult = iterative_refinement(A, b, solver=plup_solve, double_precision=False)
    ourEvenBetterResult = iterative_refinement(A, b, solver=plup_solve, double_precision=True)

    results = [
        ('NumPy solver', numpyResult),
        ('PLUP solver', ourResult),
        ('PLUP sparse solver', sparseResult),
        ('PLUP solver with iterative refinement', ourBetterResult),
        ('PLUP solver with iterative refinement and double precision', ourEvenBetterResult),
    ]
    for result in results:
        description, x = result
        print(solution_deviation(A, x, b), description)


    # Results become better only if there are just TWO iterations! WTF?!
    results = []
    for i in range(0, 4):
        x = iterative_refinement(A, b, solver=plup_solve, iterations=i, sparse=True)
        deviation = solution_deviation(A, x, b)
        print(i, deviation)
