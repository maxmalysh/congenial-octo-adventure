import numpy as np
import scipy.linalg
from pprint import pprint
from tests import test_matricies

def mprint(args, pretty=False):
    if pretty:
        pprint(args)
    else:
        print(args)

# This seems to be Partial Pivoting (частичный выбор по строке), described in the following (page 6, 3.1):
# http://archives.math.utk.edu/ICTCM/VOL08/C006/paper.pdf [1]
# This function does it all at once (not at every step of Gaussian elimination,
# as recommended in LecV.pdf page 30).
def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    m = len(M)

    # Create an identity matrix, with floating point values
    id_mat = np.matrix([[float(i == j) for i in range(m)] for j in range(m)])


    # Rearrange the identity matrix such that: the largest element of
    # first column of M is placed on the diagonal of of M. Then same thing for
    # submatrix of M, and so on.
    #
    # For example:
    #
    # 1 2 3
    # 4 5 6
    # 7 8 9
    #
    # step 0, whole matrix, 7 is the largest element of the first column, move it to diagonal (swap row 1 and 3).
    #
    # 7 8 9
    # 4 5 6   submatrix:  5 6
    # 1 2 3               2 3
    #
    # step 1, 2x2 submatrix of interest, 5 is the largest element of the first column, already on diagonal.
    #
    # step 2, 1x1 submatrix, nothing to do.
    for j in range(m):
        row = max(
            range(j, m), key=lambda i: abs(M[i, j])
            # M_ij = max from i=j to m of abs(M_ij), as in pdf [1] (page 6, 3.1)
        )
        if j != row:
            # Swap the rows
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    return np.matrix(id_mat)


def lu_decomposition(A):
    """Performs an LU Decomposition of A (which must be square)
    into PA = LU. The function returns P, L and U."""
    n = len(A)

    # Create zero matrices for L and U
    L = np.matrix([[0.0] * n for i in range(n)])
    U = np.matrix([[0.0] * n for i in range(n)])

    # Create the pivot matrix P and the multipled matrix PA
    P = pivot_matrix(A)
    PA = P @ A

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j, j] = 1.0

        # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j+1):
            s1 = sum(U[k, j] * L[i, k] for k in range(i))
            U[i, j] = PA[i, j] - s1

        # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )
        for i in range(j, n):
            s2 = sum(U[k, j] * L[i, k] for k in range(j))
            L[i, j] = (PA[i, j] - s2) / U[j, j]

    return (P, L, U)

#
# Bruhat: modified
# http://math.stackexchange.com/questions/290707/decompose-a-as-a-lpu
# https://goo.gl/UD5QqR proof by construction
# http://www.math.uiuc.edu/~mando/classes/2010F/416-manual/hw6-solutions.pdf
#
def LPU_decomposition(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    # Consider the first (from the left) nonzero entry in the first row of A.
    # Call it the pivoting entry.
    #
    # Posmultiplication by an upper triangular matrix can kill all subsequent entries in the first row
    # and make the pivoting entry equal to 1.
    #
    # Next, premultiplication by a lower triangular matrix can be used to zero all entries which are located
    # below the pivoting one in its column. Once this is done, we find a new pivoting entry, that is,
    # the first nonzero entry in the second row of the current matrix. Using postmultiplication,
    # we annihilate all entries to the right of the pivoting one in the second row,
    # and using premultiplication, then, we get rid of all entries below the pivoting one in its column, and so on.
    # In the end, we arrive at some permutational matrix P.
    #
    #
    size = A.ndim
    return np.identity(size), np.identity(size), np.identity(size)


#
# Bruhat: classic
# http://mathpar.com/ru/help/08matrix.html
#
def LPL_decomposition(A: np.matrix) -> (np.matrix, np.matrix, np.matrix):
    #  Стр. 27, матрица Q имеет единицы на побочной диагонали
    Q = np.identity(A.ndim)
    Q = Q[::-1]

    # AQ = LPU
    L, P, U = LPU_decomposition(A@Q)

    Ls = Q @ U @ Q
    return L @ (P@Q) @ Ls


A = test_matricies[0]
P, L, U = lu_decomposition(A)
#P, L, U = scipy.linalg.lu(A)

mprint("A:")
mprint(A, pretty=True)

mprint("P:")
mprint(P, pretty=True)

mprint("L:")
mprint(L, pretty=True)

mprint("U:")
mprint(U, pretty=True)

mprint("Check:")
mprint(P@L@U)



