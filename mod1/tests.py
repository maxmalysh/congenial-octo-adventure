import numpy as np
from task1golubkov import PLU_decomposition, LUP_decomposition, PLUP_decomposition

np.seterr(all='raise') # 'raise' / 'print' / 'ignore'

test_matricies = [

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

]

def test_decompositions():
    for i in range(0, len(test_matricies)):
        print("Testing matrix %d" % i)
        test_matrix = test_matricies[i]
        try:
            P, L, U = PLU_decomposition(test_matrix)
            result = P@L@U
            equal = np.allclose(test_matrix, result)

            if not equal:
                print("Failed PLU")

            L, U, P = LUP_decomposition(test_matrix)
            result = L@U@P
            equal = np.allclose(test_matrix, result)

            if not equal:
                print("Failed LUP")

            P, L, U, P_ = PLUP_decomposition(test_matrix)
            result = P @ L @ U @ P_
            equal = np.allclose(test_matrix, result)

            if not equal:
                print("Failed PLUP'")

        except Exception as e:
            print("Failed with exception on")
            print(test_matrix)
            print(e, '\n')
        finally:
            pass

print("Testing decompositions")
test_decompositions()
