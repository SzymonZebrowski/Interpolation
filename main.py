import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dir = "figs/"


def solve_equations(A, b):
    '''LU factorization with pivoting'''
    P,L,U = LU_with_pivoting(A) # działa ok
    N = len(A)


    y = np.zeros(N)
    x = np.zeros(N)
    #L = np.dot(P,L)
    #1. L * U * x = P*b
    #2. Solve L*y = P*b
    #3. Solve U*x = y
    #forward: PLy = b
    #b = np.dot(P,b)



    b = np.dot(P, b)
    y[0] = b[0] / L[0, 0]
    for i in range(1, N):
        suma = 0
        for j in range(0, i):
            suma += L[i, j] * y[j]
        print(f"Sum={suma}, b[i]={b[i]}, L[i,i]={L[i,i]}")
        y[i] = (b[i]) - suma / L[i, i]

    #backward: Ux = y
    for i in range(N - 1, -1, -1):
        suma = y[i]
        for j in range(i, N):
            if i != j:
                suma -= U[i, j] * x[j]
        x[i] = suma / U[i, i]

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


def polynomial_interpolation(data, original, filename):
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

    x_s = np.arange(min(x), max(x), 10)
    y_s = value(x_s)
    plt.plot(x, y, 'bo', label="Points")
    plot_original_profile(original)
    plt.plot(x_s, y_s, label="Interpolation")
    plt.legend(loc="best")
    plt.axis([min(x) * 0.8, max(x) * 1.2, min(y) * 0.8, max(y) * 1.2])
    plt.title(f"Polynomial interpolation - {N} points")
    plt.xlabel("Distance [m]")
    plt.ylabel("Height [m]")
    plt.savefig(f"{dir}{filename}_POLY_{N}.png")
    plt.close()


def lagrange_interpolation(data, original, filename):
    x = np.array([i[0] for i in data])[np.newaxis].T
    y = np.array([i[1] for i in data])[np.newaxis].T
    N = len(data)

    def bases(v):
        bases = []
        for i in range(N):
            base = 1
            for j in range(N):
                if(i!=j):
                    base *= ((v-x[j])/(x[i]-x[j]))
            bases.append(base)
        return bases

    def value(x):
        base = bases(x)
        val = 0
        for i in range(N):
            val += base[i] * y[i]
        return val

    x_s = np.arange(x[0, 0], x[-1, 0], 10)
    y_s = value(x_s)
    plt.plot(x, y, 'bo', label="Points")
    plot_original_profile(original)
    plt.plot(x_s, y_s, label="Interpolation")
    plt.legend(loc="best")
    plt.axis([min(x)*0.8, max(x)*1.2, min(y)*0.8, max(y)*1.2])
    plt.title(f"Lagrange interpolation - {N} points")
    plt.xlabel("Distance [m]")
    plt.ylabel("Height [m]")
    plt.savefig(f"{dir}{filename}_LAGR_{N}.png")
    plt.close()


def create_spline_matrix(data):
    x = np.array([i[0] for i in data])[np.newaxis].T
    y = np.array([i[1] for i in data])[np.newaxis].T
    n = len(data)

    N = 4 * (len(data) - 1)  # size of matrix
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
    S = np.array([[1,1,1,1]])   #1*a +1*b+1*c+1*d
    dS = np.array([[0,1,2,3]])  #0*a + 1*b +2*c + 3*d
    ddS = np.array([[0,0,2,6]]) #0*a + 0*b + 2*c + 6*d

    #regula 1 - S(x) = f(x)
    for i in range(n-1):
        h=x[i+1]-x[i]
        A[2*i, 4*i] = 1
        A[2*i + 1, 4*i:4*i+4] = np.multiply(S, [1, h, h**2, h**3])
        b[2*i] = y[i]
        b[2*i+1] = y[i+1]
    # regula 2 - S'j-1(xj)=S'j(xj) i S''j-1(xj) = S''j-1(xj)
    for i in range(1, n-1):
        h = x[i] - x[i-1]
        A[2*i + 2*(n-1) - 2, 4*(i-1):4*(i-1)+4] = np.multiply(dS, [1, 1, h, h**2])
        A[2*i + 2*(n-1) - 2, 4*i+1] = -1

        A[2*i + 2*(n-1) - 1, 4*(i-1):4*(i-1)+4] = np.multiply(ddS, [1, 1, 1, h])
        A[2*i + 2*(n-1) - 1, 4*i+2] = -2

    #regula 3 - S0''(x0) = 0 i Sn-1''(xn) = 0
    h = x[n-1]-x[n-2]
    A[4*(n-1)-2, 2] = 2
    A[4*(n-1)-1, 4*(n-2):4*(n-2)+4] = np.multiply(ddS, [1, 1, 1, h])

    return A, b


def spline_interpolation(data, original, filename):
    '''3rd degree polynomial'''
    x = np.array([i[0] for i in data])[np.newaxis].T
    y = np.array([i[1] for i in data])[np.newaxis].T
    n = len(data)

    N = 4*(len(data)-1) #size of matrix


    A, b = create_spline_matrix(data)
    c = solve_equations(A, b)

    def value(v, c):
        for i in range(n-1):
            if v >=x[i] and v<=x[i+1]:
                ca= c[4*i + 0]
                cb= c[4*i + 1]
                cc= c[4*i + 2]
                cd= c[4*i + 3]
                S = ca + cb*(v-x[i]) + cc*(v-x[i])**2 + cd*(v-x[i])**3
                return S

    x_s = np.arange(min(x), max(x), 10)
    y_s = [value(i, c) for i in x_s]
    print(y_s)
    plt.plot(x, y, 'bo', label="Points")
    plot_original_profile(original)
    plt.plot(x_s, y_s, 'g', label="Interpolation")
    plt.legend(loc="best")
    plt.axis([min(x) * 0.8, max(x) * 1.2, min(y) * 0.8, max(y) * 1.2])
    plt.title(f"Spline interpolation - {n} points")
    plt.xlabel("Distance [m]")
    plt.ylabel("Height [m]")
    plt.savefig(f"{dir}{filename}_SPLINE_{n}.png")
    plt.close()


def load_data(filename):
    #return data as list of tuples (distance, height)

    data = pd.read_csv(os.getcwd() + "/data/"+filename, sep=',')
    distance = np.array([float(x) for x in data.iloc[:, 0].values])
    altitude = np.array([float(x) for x in data.iloc[:, 1].values])

    data = [(x[0], x[1]) for x in zip(distance, altitude)]
    return data




def main():

    coords = list(range(-20, 21))
    points = [(float(x), abs(x)) for x in coords]

    #points = [(float(x), np.random.randint(-10, 10)) for x in range(-10, 20, 4)]
    #points = [(1., 3.), (3., 7), (8., 10.)]
    #points = [(1., 6.), (3., -2.), (5., 4.)]
    #points = [(1., 1.), (2., 8.), (3., 4.), (4., 1.)]
    points = [(0., 4.), (2., 1.), (3., 6.), (4., 1.)]




    x = np.array([i[0] for i in points])[np.newaxis].T
    y = np.array([i[1] for i in points])[np.newaxis].T
    #spline_interpolation(points)
    lagrange_interpolation(points)
    #polynomial_interpolation(points)


def plot_original_profile(data):
    distance = [x[0] for x in data]
    height = [x[1] for x in data]
    plt.plot(distance, height, label="Profile")
    return plt

if __name__ == "__main__":
    names = ["WielkiKanionKolorado", "MountEverest"]
    filenames = [(name + '.csv.') for name in names]
    nums = [32,16,8,4,2,1]
    for i in range(len(names)):
        org_data = load_data(filenames[i])

        for j in nums:
            data = org_data[0::j]
            lagrange_interpolation(data, org_data, names[i])
            spline_interpolation(data, org_data, names[i])

