import unittest
import numpy as np
from scipy.linalg import solve as npsolve
from task3 import plu_solve

class NumberedTest(unittest.TestCase):
    test_number = 0

    def __init__(self):
        super().__init__()
        NumberedTest.test_number += 1
        self.test_number = SolverTest.test_number


class SolverTest(NumberedTest):
    def __init__(self, matrix, vector):
        super().__init__()
        self.matrix = matrix
        self.vector = vector

    def __str__(self):
        return self.__class__.__name__ + ":" + str(self.test_number)


class Task3Test(SolverTest):
    def runTest(self):
        trueResult = npsolve(self.matrix, self.vector)
        ourResult = plu_solve(self.matrix, self.vector)
        equal = np.allclose(ourResult, trueResult)
        self.assertTrue(equal, msg="Failed on \nA:\n{0},\nb:\n{1}".format(self.matrix, self.vector))


def get_random_suite():
    suite = unittest.TestSuite()
    test_types = [
        Task3Test,
    ]

    msize = 30
    for i in range(0, 250):
        singular = True

        while singular:
            matrix = np.random.randint(low=-100, high=100, size=(msize, msize))
            if np.linalg.det(matrix) != 0:
                singular = False

        vector = np.random.randint(low=-100, high=100, size=msize)
        tests = [test(matrix, vector) for test in test_types]
        suite.addTests(tests)

    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(get_random_suite())
