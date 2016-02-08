import unittest

import numpy as np

from utils import MatrixBuilder
from task1 import PLU_decomposition, LUP_decomposition, PLUP_decomposition, Sparse_decomposition
from task1 import lpl_decompose, lpu_decompose
from utils import test_matrices

np.seterr(all='raise') # 'raise' / 'print' / 'ignore'

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


def get_random_suite(test_number=50, msize=50):
    suite = unittest.TestSuite()
    test_types = [
        PLU_Test,
        LUP_Test,
        PLUP_Test,
    ]

    for i in range(0, test_number):
        matrix = MatrixBuilder(msize).gen()
        tests = [ test(matrix, i) for test in test_types ]
        suite.addTests(tests)

    return suite

def get_random_nonsingular_suite(test_number=50, msize=50):
    suite = unittest.TestSuite()
    test_types = [
        LPU_Test,
        LPL_Test,
    ]

    for i in range(0, test_number):
        matrix = MatrixBuilder(msize).nonsingular().gen()
        tests = [test(matrix, i) for test in test_types]
        suite.addTests(tests)

    return suite

def get_random_sparse_nonsingular_suite(test_number=10, msize=15):
    suite = unittest.TestSuite()

    for i in range(0, test_number):
        matrix = MatrixBuilder(msize).nonsingular().dok().randsparse().gen()
        tests = [Sparse_Test(matrix, i)]
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
        ]
        suite.addTests(tests)

    return suite


if __name__ == '__main__':
    combo_suite = unittest.TestSuite([
        # get_suite(),
        # get_random_suite(),
        # get_random_nonsingular_suite(),
        # get_random_sparse_nonsingular_suite(),
        get_random_sparse_nonsingular_suite(test_number=3000, msize=3),
    ])
    unittest.TextTestRunner(verbosity=2).run(combo_suite)