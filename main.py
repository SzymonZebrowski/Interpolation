import numpy as np
import scipy


def solve_equations(A, b):
    '''Try Gauss-Seidl method, if iterations > 1000, then do LU factorization'''
    N = len(A)
    x_n = np.ones((N, 1))
    x = np.ones((N, 1))
    iter = 0
    bound = 1e-9
    while np.linalg.norm(np.dot(A, x) - b) > bound:
        for i in range(N):
            p1 = np.dot(A[i, :i], x_n[:i])
            p2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_n[i, 0] = (b[i] - p1 - p2) / A[i, i]
        x = np.copy(x_n)
        iter += 1

    norm_res = np.linalg.norm(np.dot(A, x) - b)

    if iter < 1000:
        return x

    U = np.copy(A)
    L = np.eye(N)

    # decomposition

    for k in range(N - 1):
        for j in range(k + 1, N):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:N] -= L[j, k] * U[k, k:N]

    # solving
    # forward substitution Ly = b
    y = np.zeros(N)
    y[0] = b[0] / L[0, 0]
    # y[0, 0] = self.b[0, 0] / L[0, 0]
    for i in range(1, N):
        sum = 0
        for j in range(0, i):
            sum += L[i, j] * y[j]
        y[i] = (b[i]) - sum / L[i, i]

    # backward substitution Ux = y

    for i in range(N - 1, -1, -1):
        sum = y[i]
        for j in range(i, N):
            if i != j:
                sum -= U[i, j] * x[j]
        x[i] = sum / U[i, i]

    norm_res = np.linalg.norm(np.dot(A, x) - b)
    return x


def polynomial_interpolation(data):
    

def main():
    A = np.array([3, 2, -1, 2, -2, 4, -1, 0.5, -1]).reshape((3, 3))
    b = np.array([1., -2., 0.])[np.newaxis].T

    print(A, b)
    print(solve_equations(A, b))

    print(np.linalg.solve(A,b))
if __name__ == "__main__":
    main()