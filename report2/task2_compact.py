import numpy as np

ITERATION_LIMIT = 1000

# initialize the matrix
A = np.array([[10., -1.,  2.,  0.],
              [-1., 11., -1.,  3.],
              [2.,  -1., 10., -1.],
              [0.,   3., -1.,  8.]])

A2 = np.array(
      [[ 77., -23., -32.,   0.,   0.],
       [-23.,  53.,   0.,   0., -18.],
       [-32.,   0.,  90.,  -5.,   0.],
       [  0.,   0.,  -5.,  49., -15.],
       [  0., -18.,   0., -15.,  89.]])

# initialize the RHS vector
b = np.array([6., 25., -11., 15.])
b2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

def jacobi_method(A: np.ndarray, b: np.ndarray):
    x = np.zeros_like(b)

    for it_count in range(ITERATION_LIMIT):
        print("%2d. Current solution: %s" % (it_count, x))
        x_new = np.zeros_like(x)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.allclose(x, x_new, atol=1e-8):
            break

        x = x_new

    return x

def gauss_seidel(A: np.ndarray, b: np.ndarray):
    x = np.zeros_like(b)

    for it_count in range(ITERATION_LIMIT):
        print("%2d. Current solution: %s" % (it_count, x))
        x_new = np.zeros_like(x)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.allclose(x, x_new, rtol=1e-8):
            break

        x = x_new

    return x

def sor_method(A: np.ndarray, b: np.ndarray, w=1.0):
    x = np.zeros_like(b)

    for it_count in range(ITERATION_LIMIT):
        print("%2d. Current solution: %s" % (it_count, x))
        x_new = np.zeros_like(x)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1.0 - w)*x[i] + w * (b[i] - s1 - s2) / A[i, i]

        if np.allclose(x, x_new, rtol=1e-8):
            break

        x = x_new

    return x

def ssor_method(A: np.ndarray, b: np.ndarray, w=1.0):
    x = np.zeros_like(b)
    xk = np.zeros(shape=(ITERATION_LIMIT, x.shape[0]), dtype=np.float)

    for it_count in range(ITERATION_LIMIT):
        print("%2d. Current solution: %s" % (it_count, x))
        k = it_count
        xk[k] = np.zeros_like(x)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], xk[k-1][:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            xk[k/2][i] = (1.0 - w)*x[i] + w * (b[i] - s1 - s2) / A[i, i]
        #
        for i in reversed(range(A.shape[0])):
            s1 = np.dot(A[i, :i], xk[k/2][:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            xk[k][i] = (1.0 - w)*x[i] + w * (b[i] - s1 - s2) / A[i, i]

        if np.allclose(x, xk[k], rtol=1e-8):
            break

        x = xk[k]

    return x

#A, b = A2, b2
x = sor_method(A, b)
print("Final solution:")
print(x)

error = np.dot(A, x) - b
print("Error:")
print(error)