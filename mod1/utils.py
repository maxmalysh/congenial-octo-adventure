import numpy as np
import random
from scipy import sparse

def get_random_matrix(n):
    matrix = np.random.randint(low=-100, high=100, size=(30,30))
    return matrix.astype(np.float)

def get_nonsingular_matrix(n):
    singular = True

    while singular:
        matrix = np.random.randint(low=-100, high=100, size=(n, n))
        if np.linalg.det(matrix) != 0:
            singular = False

    return matrix.astype(np.float)

def get_arrow_matrix(n):
    singular = True

    while singular:
        matrix = sparse.dok_matrix((n, n), dtype=np.float)

        for i in range(0, n):
            matrix[0, i] = random.randint(-100, 100)
            matrix[i, 0] = random.randint(-100, 100)
            matrix[i, i] = random.randint(-100, 100)

        if np.linalg.det(matrix.todense()) != 0:
            singular = False

    return matrix

def get_random_sparse_matrix(n):
    singular = True

    while singular:
        matrix = sparse.dok_matrix((n, n), dtype=np.float)
        for rand_iterations in range(0, int((n ** 2) / 4)):
            matrix[random.randint(0, n - 1), random.randint(0, n - 1)] = random.randint(-100, 100)
        if np.linalg.det(matrix.todense()) != 0:
            singular = False

    return matrix


test_matrices = [

    np.matrix([
        [7, 3, -1, 2],
        [3, 8, 1, -4],
        [-1, 1, 4, -1],
        [2, -4, -1, 6],
    ]),

    np.matrix([
        [0,  1, 0],
        [-8, 8, 1],
        [2, -2, 0],
    ]),

    # LUP / PLU, p.3: http://www.math.unm.edu/~loring/links/linear_s08/LU.pdf
    np.matrix([
        [2, 1, 0, 1],
        [2, 1, 2, 3],
        [0, 0, 1, 2],
        [-4, -1, 0, -2],
    ]),

    # Для Брюа: http://mathpar.com/ru/help/08matrix.html
    np.matrix([
        [1, 4, 0, 1],
        [4, 5, 5, 3],
        [1, 2, 2, 2],
        [3, 0, 0, 1],
    ]),

    np.matrix([
        [0, 2],
        [1, 4],
    ]),

    np.matrix([
        [2, -2, 0],
        [0,  1, 0],
        [-8, 8, 1],
    ]),

    np.matrix([
        [1, 3, 7, 2, 2],
        [2, 1, 9, 8, 3],
        [7, 8, 5, 1, 3],
        [0, 8, 2, 6, 3],
        [0, 3, 2, 2, 2],
    ]),

    np.matrix([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ])
]