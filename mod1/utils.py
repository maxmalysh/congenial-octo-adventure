import numpy as np
import random
from scipy import sparse

default_type = np.float

def get_random_vector(n, low=-100, high=100):
    vector = np.random.randint(low=low, high=high, size=n)
    return vector.astype(default_type)

class MatrixBuilder:
    def __init__(self, size, low=-100, high=100, dtype=default_type):
        self.matrix = np.zeros((size, size), dtype=dtype)
        self.size = size
        self.low = low
        self.high = high
        self.fill_element = lambda: np.random.randint(low=low, high=high)
        self.fill_matrix = self._get_default_filler()

    def nonsingular(self):
        self.nonsingular = True
        return self

    def dok(self):
        self.matrix = sparse.dok_matrix(self.matrix)
        return self

    def arrow(self):
        def filler():
            for i in range(0, self.size):
                self.matrix[0, i] = self.fill_element()
                self.matrix[i, 0] = self.fill_element()
                self.matrix[i, i] = self.fill_element()
        self.fill_matrix = filler
        return self

    def randsparse(self):
        def filler():
            for rand_iterations in range(0, int((self.size ** 2) / 4)):
                i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                self.matrix[i, j] = random.randint(self.low, self.high)
        self.fill_matrix = filler
        return self

    def nearsingular(self, el_n=None):
        if el_n == None:
            el_n = self.size

        def filler():
            mult = np.finfo(default_type).eps * 1000.0

            for i in range(0, self.size):
                self.matrix[i, i] = random.randint(-self.low, self.high)

            for n in range(0, el_n):
                i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                self.matrix[i, j] = random.randint(self.low, self.high) * mult

        self.fill_matrix = filler
        return self

    def stieltjes(self):
        def filler():
            positive_definite = False

            while not positive_definite:
                for i in range(0, self.size):
                    for j in range(0, self.size):
                        high = 0 if i != j else self.high
                        value = np.random.randint(self.low, high=high)
                        self.matrix[i, j] = value
                        self.matrix[j, i] = value

                try:
                    np.linalg.cholesky(self.matrix)
                    positive_definite = True
                except np.linalg.LinAlgError as e:
                    positive_definite = False

        self.fill_matrix = filler
        return self

    def sparse_stieltjes(self, extra_elements=None):
        if extra_elements == None:
            extra_elements = self.size

        def filler():
            positive_definite = False

            while not positive_definite:
                self.matrix = np.zeros(dtype=default_type, shape=(self.size, self.size))
                for i in range(0, self.size):
                    self.matrix[i, i] =  np.random.randint(self.low, high=self.high)

                for n in range(0, int(extra_elements / 2)):
                    i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                    value = random.randint(self.low, 0)
                    self.matrix[i, j] = value
                    self.matrix[j, i] = value

                try:
                    np.linalg.cholesky(self.matrix)
                    positive_definite = True
                except np.linalg.LinAlgError as e:
                    positive_definite = False

        self.fill_matrix = filler
        return self

    def hilbert(self):
        def filler():
            for i in range(0, self.size):
                for j in range(0, self.size):
                    self.matrix[i, j] = 1.0 / (i+j+1)

        self.fill_matrix = filler
        return self

    def _get_default_filler(self):
        def filler():
            for i in range(0, self.size):
                for j in range(0, self.size):
                    self.matrix[i, j] = self.fill_element()
        return filler

    def _singular(self):
        if isinstance(self.matrix, sparse.base.spmatrix):
            return np.linalg.det(self.matrix.todense()) == 0
        return np.linalg.det(self.matrix) == 0

    def gen(self):
        if self.nonsingular:
            while self._singular():
                self.fill_matrix()
        else:
            self.fill_matrix()
        return self.matrix


test_matrices = [

    np.matrix([
        [790000, 3, -1, 2],
        [33, 8, 15, -4],
        [-1, 2, 4, -1212312],
        [20, -4, -10, 6],
    ], dtype=default_type),

    np.matrix([
        [0,  1, 0],
        [-8, 8, 1],
        [2, -2, 0],
    ],  dtype=default_type),

    # LUP / PLU, p.3: http://www.math.unm.edu/~loring/links/linear_s08/LU.pdf
    np.matrix([
        [2, 1, 0, 1],
        [2, 1, 2, 3],
        [0, 0, 1, 2],
        [-4, -1, 0, -2],
    ],  dtype=default_type),

    # Для Брюа: http://mathpar.com/ru/help/08matrix.html
    np.matrix([
        [1, 4, 0, 1],
        [4, 5, 5, 3],
        [1, 2, 2, 2],
        [3, 0, 0, 1],
    ],  dtype=default_type),

    np.matrix([
        [0, 2],
        [1, 4],
    ],  dtype=default_type),

    np.matrix([
        [2, -2, 0],
        [0,  1, 0],
        [-8, 8, 1],
    ],  dtype=default_type),

    np.matrix([
        [1, 3, 7, 2, 2],
        [2, 1, 9, 8, 3],
        [7, 8, 5, 1, 3],
        [0, 8, 2, 6, 3],
        [0, 3, 2, 2, 2],
    ],  dtype=default_type),

    np.matrix([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ],  dtype=default_type),
]
