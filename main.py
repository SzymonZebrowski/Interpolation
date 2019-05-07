import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def solve_equations(A, b):
    '''LU factorization with pivoting'''
    M,N = np.shape(A)
    P,L,U = LU_with_pivoting(A) # działa ok

    # x = A \ B
    # x = inv(A) * B
    # x = U \ (L \ (P*b))
    x = np.linalg.lstsq(U, (np.linalg.lstsq(L, (np.dot(P, b)))[0]))[0]
    return x


def LU_with_pivoting(A):
    M, N = np.shape(A)
    U = np.copy(A)
    L = np.eye(N)
    P = np.eye(N)

    # decomposition
    for k in range(M - 1):
        # pivoting
        pivot = max(abs(U[k:M, k]))
        for i in range(k, M):
            if abs(U[i, k]) == pivot:
                ind = i
                break
        #change rows
        U[[k, ind], k:M] = U[[ind, k], k:M]
        L[[k, ind], :k] = L[[ind, k], :k]
        P[[k, ind], :] = P[[ind, k], :]

        for j in range(k+1, M):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:M] -= L[j, k] * U[k, k:M]

    return P, L, U

def polynomial_interpolation(data):
    x = np.array([i[0] for i in data])[np.newaxis].T
    y = np.array([i[1] for i in data])[np.newaxis].T
    N = len(data)

    #Vandermonde matrix
    V = np.array([[i**n for n in range(N-1, -1, -1)] for i in x]).reshape((N, N))

    a = solve_equations(V, y)
    powers = list(reversed(range(N)))

    def value(x):
        x_n = [x**n for n in powers]
        vals = []
        for i in range(N):
            vals.append(x_n[i]*a[i])
        return sum(vals)

    x_s = np.arange(min(x), max(x), 0.01)
    y_s = value(x_s)
    plt.plot(x, y, 'bo')
    plt.plot(x_s, y_s)
    plt.title("Polynomial interpolation")
    plt.show()

def lagrange_interpolation(data):
    x = np.array([i[0] for i in data])[np.newaxis].T
    y = np.array([i[1] for i in data])[np.newaxis].T
    N = len(data)

    def bases(v):
        bases = []
        for i in range(N):
            base = 1
            for j in range(N):
                if(i!=j):
                    base*=(v-x[j])/(x[i]-x[j])
            bases.append(base)
        return bases

    def value(x):
        base = bases(x)
        vals = []
        for i in range(N):
            vals.append(base[i] * y[i])
        return sum(vals)

    x_s = np.arange(min(x), max(x), 0.01)
    y_s = value(x_s)
    plt.plot(x, y, 'bo')
    plt.plot(x_s, y_s)
    plt.title("Lagrange interpolation")
    plt.show()


def main():
    coords = list(range(-2,3))
    print(coords)
    points = [(float(x), float(abs(x))) for x in coords]
    #points = [(1., 3.), (3., 7), (8., 10.)]
    #points = [(1., 1.), (2., 2.), (3., 1.), (4., 1.), (0., 0.)]
    #points = [(0., 4.), (2., 1.), (3., 6.), (4., 1.)]


    polynomial_interpolation(points)
    lagrange_interpolation(points)


if __name__ == "__main__":
    main()