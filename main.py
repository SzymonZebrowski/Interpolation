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
    N = len(data)

    no_splits = N-1
    results = []
    for i in range(no_splits-1):
        #Si - odcinek między punktami xi, a xi+1
        #h = xi+1 - xi
        h=x[i+1]-x[i]
        A = np.array([[1,   0,   0,   0,   0,   0,   0,   0],
                      [1,   h,h**2,h**3,   0,   0,   0,   0],
                      [0,   0,   0,   0,   1,   0,   0,   0],
                      [0,   0,   0,   0,   1,   h,h**2,h**3],
                      [0,   1, 2*h,3*h**2, 0,  -1,   0,   0],
                      [0,   0,   2, 6*h,   0,   0,  -2,   0],
                      [0,   0,   1,   0,   0,   0,   0,   0],
                      [0,   0,   0,   0,   0,   0,   2, 6*h]], dtype=np.float64)
        b = np.array([y[i,0], y[i+1], y[i+1], y[i+2], 0, 0, 0, 0])[np.newaxis].T
        c = solve_equations(A, b)
        print("=============================")
        print(c)
        results.append(c)

    splits = []
    for j in results:
        for i in range(no_splits):
            splits.append(j[i*4:i*4+4, 0])

    print("Splits: ", splits)

    def value(v, split, i):
        a= split[i][0]
        b= split[i][1]
        c= split[i][2]
        d= split[i][3]
        S = a + b*(v-x[i]) + c*(v-x[i])**2 + d*(v-x[i])**3
        return S

    x_s = np.arange(x[0], x[2], 0.01)
    y_s1 = value(np.arange(x[0], x[1], 0.01), splits, 0)
    y_s2 = value(np.arange(x[1], x[2], 0.01), splits, 1)
    print(y_s1)
    print(y_s2)
    y_s = list(y_s1)+list(y_s2)
    plt.plot(x, y, 'bo')
    plt.plot(x_s, y_s)
    plt.title("Spline interpolation")
    plt.show()


def main():
    coords = list(range(-2, 3))
    points = [(float(x), float(abs(x))) for x in coords]
    points = [(1., 3.), (3., 7), (8., 10.)]
    points = [(1.,6.), (3.,-2.), (5.,4.)]
    points = [(1., 1.), (2., 8.), (3., 4.), (4., 1.)]
    #points = [(0., 4.), (2., 1.), (3., 6.), (4., 1.)]

    spline_interpolation(points)


if __name__ == "__main__":
    main()