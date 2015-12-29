import numpy as np
import scipy.linalg
from pprint import pprint


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

]

def mprint(args, pretty=False):
    if pretty:
        pprint(args)
    else:
        print(args)


def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    m = len(M)

    # Create an identity matrix, with floating point values
    id_mat = np.matrix([[float(i == j) for i in range(m)] for j in range(m)])

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of of M
    for j in range(m):
        row = max(
            range(j, m), key=lambda i: abs(M[i, j])
        )
        if j != row:
            # Swap the rows
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    # return np.matrix([
    #     [0, 1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1],
    # ])

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



