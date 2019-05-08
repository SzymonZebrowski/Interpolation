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

def spline_interpolation(data):
    '''degree 3 polynomial'''
    x = np.array([i[0] for i in data])[np.newaxis].T
    y = np.array([i[1] for i in data])[np.newaxis].T
    n = len(data)

    N = 4*(len(data)-1) #size of matrix
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
        #Si - odcinek między punktami xi, a xi+1
        #h = xi+1 - xi
        #A = np.array([[1,   0,   0,   0,   0,   0,   0,   0],
        #              [1,   h,h**2,h**3,   0,   0,   0,   0],
        #              [0,   0,   0,   0,   1,   0,   0,   0],
        #              [0,   0,   0,   0,   1,   h,h**2,h**3],
        #              [0,   1, 2*h,3*h**2, 0,  -1,   0,   0],
        #              [0,   0,   2, 6*h,   0,   0,  -2,   0],
        #              [0,   0,   1,   0,   0,   0,   0,   0],
        #              [0,   0,   0,   0,   0,   0,   2, 6*h]], dtype=np.float64)
        #b = np.array([y[i,0], y[i+1], y[i+1], y[i+2], 0, 0, 0, 0])[np.newaxis].T

    A[0,0] = 1.0
    b[0] = y[0]
    A[1,2] = 2.0

    h = x[1]-x[0]
    A[1,1] = 1.0
    A[1,2] = 2*h
    A[1,3] = 3*h*h
    A[1,5] = -1.

    for i in range(1, n-1):
        h = (x[i] - x[i - 1]);

        A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 0] = 1.0
        A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 1] = h
        A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 2] = h * h
        A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 3] = h * h * h
        b[2 + 4 * (i - 1) + 0] = y[i]

        A[2 + 4 * (i - 1) + 1][4 * i + 0] = 1.0
        b[2 + 4 * (i - 1) + 1] = y[i]

        A[2 + 4 * (i - 1) + 2][4 * (i - 1) + 1] = 1.0
        A[2 + 4 * (i - 1) + 2][4 * (i - 1) + 2] = 2 * h
        A[2 + 4 * (i - 1) + 2][4 * (i - 1) + 3] = 3 * h * h
        A[2 + 4 * (i - 1) + 2][4 * i + 1] = -1.0
        b[2 + 4 * (i - 1) + 2] = 0.0

        A[2 + 4 * (i - 1) + 3][4 * (i - 1) + 2] = 2.0
        A[2 + 4 * (i - 1) + 3][4 * (i - 1) + 3] = 6.0 * h
        A[2 + 4 * (i - 1) + 3][4 * i + 2] = -2.0
        b[2 + 4 * (i - 1) + 3] = 0.0

        for i in range(n-1, n):
            h = (x[i] - x[i - 1]);

            A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 0] = 1.0
            A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 1] = h
            A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 2] = h * h
            A[2 + 4 * (i - 1) + 0][4 * (i - 1) + 3] = h * h * h
            b[2 + 4 * (i - 1) + 0][0] = y[i]

            A[2 + 4 * (i - 1) + 1][4 * (i - 1) + 2] = 2.0
            A[2 + 4 * (i - 1) + 1][4 * (i - 1) + 3] = 6.0 * h
            A[2 + 4 * (i - 1) + 1][0] = 0

    print(A)
    print(b)
    c = solve_equations(A, b)

    print(c)

    def value(v,c):
        for i in range(n-1):
            if v >=x[i] and v<=x[i+1]:
                ca= c[4*i +0]
                cb= c[4*i +1]
                cc= c[4*i +2]
                cd= c[4*i +3]
                S = ca + cb*(v-x[i]) + cc*(v-x[i])**2 + cd*(v-x[i])**3
                return S

    x_s = np.arange(min(x), max(x), 0.01)
    y_s = [value(i, c) for i in x_s]
    plt.plot(x, y, 'bo')
    plt.plot(x_s, y_s)
    plt.title("Spline interpolation")
    plt.show()


def main():
    coords = list(range(-2, 10))
    #points = [(float(x), float(abs(x))) for x in coords]
    #points = [(1., 3.), (3., 7), (8., 10.)]
    points = [(1., 6.), (3., -2.), (5., 4.)]
    #points = [(1., 1.), (2., 8.), (3., 4.), (4., 1.)]
    #points = [(0., 4.), (2., 1.), (3., 6.), (4., 1.)]

    f = lambda x: 2*x**3 - 25*x**2 - 8*x + 11
    #points = [(i, f(i)) for i in range(-2, 10)]

    x = np.array([i[0] for i in points])[np.newaxis].T
    y = np.array([i[1] for i in points])[np.newaxis].T
    spline_interpolation(points)
    lagrange_interpolation(points)

    plt.plot(np.arange(min(x), max(x), 0.01), f(np.arange(min(x), max(x), 0.01)))
    plt.title("Original function")
    plt.show()



if __name__ == "__main__":
    main()