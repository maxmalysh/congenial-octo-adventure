import numpy as np
import utils
import matplotlib.pyplot as plt
from scipy.linalg import solve as npsolve
from scipy.sparse.base import spmatrix
from collections import defaultdict
from utils import MatrixBuilder
from task1 import PLUP_decomposition, Sparse_decomposition

norm = lambda A: np.linalg.norm(A, ord=np.inf)
norm1 = lambda A: np.linalg.norm(A, ord=1)

def relative_residual(A, x, b):
    r = b - A.dot(x)
    return norm(r) / norm(b)

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


def iterative_refinement(A: np.matrix, b: np.array, solver,
                         iterations=1, double_precision=False,
                         sparse=False) -> np.array:
    x = solver(A, b) if not sparse else solver(A, b, sparse=True)

    for i in range(0, iterations):
        chosen_type = np.longfloat if double_precision else A.dtype
        r = b.astype(chosen_type) - A.astype(chosen_type).dot(x).astype(chosen_type)
        z = solver(A, r.astype(A.dtype))
        x = x + z

    return x


def solution_deviation(A:np.matrix, x:np.array, b:np.array):
    differences = b - A.dot(x)
    return np.mean( [abs(err) for err in differences] )


def generate_test_system(size=100, sparse=False):
    vector = utils.get_random_vector(size, low=-100000, high=100000)

    if not sparse:
        # matrix = MatrixBuilder(size, low=-100000, high=100000).nonsingular().gen()
        matrix = MatrixBuilder(size).nonsingular().nearsingular().gen()
    else:
        matrix = MatrixBuilder(size).nonsingular().randsparse().gen()

    return matrix, vector

