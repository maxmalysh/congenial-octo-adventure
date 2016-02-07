import unittest
import numpy as np
import random
from scipy import sparse
from task1 import PLU_decomposition, LUP_decomposition, PLUP_decomposition, Sparse_decomposition
from task1 import lpl_decompose, lpu_decompose

np.seterr(all='raise') # 'raise' / 'print' / 'ignore'

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


class DecompositionTest(unittest.TestCase):
    def __init__(self, matrix, matrix_number):
        super().__init__()
        self.matrix = matrix
        self.matrix_number = matrix_number

    def __str__(self):
        return self.__class__.__name__ + ":" + str(self.matrix_number)

class PLU_Test(DecompositionTest):
    def runTest(self):
        P, L, U = PLU_decomposition(self.matrix)
        result = P@L@U
        equal = np.allclose(self.matrix, result)
        self.assertTrue(equal)

class LUP_Test(DecompositionTest):
    def runTest(self):
        L, U, P = LUP_decomposition(self.matrix)
        result = L@U@P
        equal = np.allclose(self.matrix, result)
        self.assertTrue(equal)

class PLUP_Test(DecompositionTest):
    def runTest(self):
        P, L, U, P_ = PLUP_decomposition(self.matrix)
        result = P @ L @ U @ P_
        equal = np.allclose(self.matrix, result)
        self.assertTrue(equal)

class LPU_Test(DecompositionTest):
    def runTest(self):
        L, P, U = lpu_decompose(self.matrix)
        result = L @ P @ U
        equal = np.allclose(self.matrix, result)
        self.assertTrue(equal)

class LPL_Test(DecompositionTest):
    def runTest(self):
        L, P, L_ = lpl_decompose(self.matrix)
        result = L @ P @ L_
        equal = np.allclose(self.matrix, result)
        self.assertTrue(equal)


class Sparse_Test(DecompositionTest):
    def runTest(self):
        P, L, U, P_ = Sparse_decomposition(self.matrix)
        result = P @ L @ U @ P_
        equal = np.allclose(self.matrix.todense(), result)
        self.assertTrue(equal)


def get_random_suite():
    suite = unittest.TestSuite()
    test_types = [
        PLU_Test,
        LUP_Test,
        PLUP_Test,
    ]

    for i in range(0, 20):
        matrix = np.random.randint(low=-100, high=100, size=(30,30))
        matrix = matrix.astype(np.float)
        tests = [ test(matrix, i) for test in test_types ]
        suite.addTests(tests)

    return suite


def get_random_nonsingular_suite():
    suite = unittest.TestSuite()
    test_types = [
        LPU_Test,
        LPL_Test,
    ]

    for i in range(0, 100):
        singular = True

        while singular:
            matrix = np.random.randint(low=-100, high=100, size=(50,50))
            matrix = matrix.astype(np.float)
            if np.linalg.det(matrix) != 0:
                singular = False

        tests = [test(matrix, i) for test in test_types]
        suite.addTests(tests)

    return suite


def get_random_sparse_nonsingular_suite():
    suite = unittest.TestSuite()

    for i in range(0, 10):
        singular = True

        while singular:
            n = 50
            matrix = sparse.dok_matrix((n, n), dtype=np.float)
            for rand_iterations in range(0, int((n ** 2) / 4)):
                matrix[random.randint(0, n - 1), random.randint(0, n - 1)] = random.randint(-100, 100)
            if np.linalg.det(matrix.todense()) != 0:
                singular = False

        tests = [Sparse_Test(matrix, i)]  # PLUP_Test(matrix.todense(), i),
        suite.addTests(tests)

    return suite

def get_suite():
    suite = unittest.TestSuite()

    for i in range(0, len(test_matrices)):
        matrix = test_matrices[i]
        tests = [
            PLU_Test(matrix, i),
            LUP_Test(matrix, i),
            PLUP_Test(matrix, i),
            LPU_Test(matrix.copy(), i),
            LPL_Test(matrix, i),
        ]
        suite.addTests(tests)

    return suite


if __name__ == '__main__':
    # unittest.TextTestRunner(verbosity=2).run(get_random_suite())
    # unittest.TextTestRunner(verbosity=2).run(get_random_nonsingular_suite())
    unittest.TextTestRunner(verbosity=2).run(get_random_sparse_nonsingular_suite())
