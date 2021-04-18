"""Programme TP1 Ma223 @AdrienRicord @VincentLenoble"""

#Initialisation...

import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import time

A = np.array([[4,-2,-4], [-2,10,5], [-4,5,6]])
B = np.array([[6],[-9],[-7]])

#Fonctions...

def ReductionGauss(Aaug):

    n,m = Aaug.shape

    for i in range(0,n-1):
        for k in range(i+1,n):
            Aaug[k,:] = Aaug[k,:]-(Aaug[k,i]/Aaug[i,i])*Aaug[i,:]

    return Aaug

def ResolutionSystTriInf(Taug):

    T = np.copy(Taug)
    n,m = T.shape
    y = np.zeros(n)
      
    for i in range(n):
        y[i] = (T[i,n] - np.sum(T[i,:i]*y[:i]))/T[i,i]
                    
    return y

def ResolutionSystTriSup(Taug):

    T = np.copy(Taug)
    n,m = T.shape
    x = np.zeros(n)
      
    for i in range(n-1,-1,-1):
        x[i] = (T[i,n] - np.sum(T[i,i+1:n]*x[i+1:n]))/T[i,i]
                    
    return x

def Gauss(A,B):

    Aaug = np.concatenate((A,B), axis=1)
    Taug = ReductionGauss(Aaug)
    X = ResolutionSystTriSup(Taug)

    return X

def DecompositionLU(A):

    n = np.shape(A)[0]
    L = np.identity(n)
    U = np.copy(A)

    for i in range(0,n-1):
        for k in range(i+1,n):
            L[k,i] = U[k,i]/U[i,i]
            U[k,:] = U[k,:]-(L[k,i])*U[i,:]

    return L,U

def ResolutionLU(L,U,B):

    n,m = L.shape
    Y = np.zeros(n)

    for i in range(n):
        Y[i] = B[i] - sum(L[i,:i]*Y[:i])

    X = np.zeros(n)

    for i in range(n-1,-1,-1):
        X[i] = (Y[i] - sum(U[i,i:n]*X[i:n]))/U[i,i]

    return X

def DecompositionCholesky(A):

    n,m = A.shape
    L = np.zeros((n,n))
    for i in range (0,n):
        for k in range (0,n):
            if i==k :
                L[i,k] = sqrt(A[i,k] - np.sum(L[i,:i]**2))
            if i > k :
                L[i,k] = (A[i,k] - np.sum(L[i,:i]*L[k,:i])) / L[k,k]
    return L

def ResolutionCholesky(L,B):

    LT = np.transpose(L)

    Laug = np.concatenate((L,B), axis=1)
    Y = ResolutionSystTriInf(Laug)
    Y = Y.reshape(-1,1)

    LTaug = np.concatenate((LT,Y), axis=1)
    X = ResolutionSystTriSup(LTaug)

    return X

def MatSym(A): #fonction permettant de transformer une matrice quelconque en matrice symétrique
    
    n,m = A.shape
    for m in range(0,n):
        for i in range(1,n):
            A[i,m] = A[m,i]

    return(A)

def MatCho(Asym): #fonction qui prend en paramètre une matrice symétrique et qui en fait une matrice adaptée pour Cholesky

    At = np.transpose(Asym)
    Aw = np.matmul(At,Asym)

    return(Aw)

#Programme...

print("\n-------------Décomposition de Cholesky---------------\n")

start = time.time()
L = DecompositionCholesky(A)
print("La matrice L :\n", L)

X = ResolutionCholesky(L,B)
print("La matrice solution :\n", X)
end = time.time()

t = end - start
print("Temps d'exécution : ", t, "s")

e = np.linalg.norm(np.dot(A,X)-np.ravel(B))
print("Erreur Cholesky", e)

print("\n--------------------Graphiques-----------------------\n")

test = []

for i in range (10,511,50): #de ... à ... avec un pas de ...
    test.append(i)

len_test = len(test)
Erreur_LU = []
Erreur_Cho = []
Erreur_ChoLinalg = []
Erreur_Linalg = []

Temps_LU = []
Temps_Cho = []
Temps_ChoLinalg = []
Temps_Linalg = []

for i in range(len_test):
    
    n = test[i]
    A = np.random.random(size=(n,n))

    #création de la matrice de travail Aw...
    Asym = MatSym(A)
    Aw = MatCho(Asym)

    B = np.random.random(size=(n,1))

    start = time.time()
    L,U = DecompositionLU(Aw)
    X1 = ResolutionLU(L,U,B)
    LU = np.matmul(L,U)
    end = time.time()
    E1 = np.linalg.norm(np.dot(Aw,X1)-np.ravel(B))
    Erreur_LU.append(E1)
    Temps_LU.append(end - start)

    start = time.time()
    A2 = DecompositionCholesky(Aw)
    X2 = ResolutionCholesky(A2,B)
    end = time.time()
    E2 = np.linalg.norm(np.dot(Aw,X2)-np.ravel(B))
    Erreur_Cho.append(E2)
    Temps_Cho.append(end - start)

    start = time.time()
    MatChoLinalg = np.linalg.cholesky(Aw)
    X3 = ResolutionCholesky(MatChoLinalg,B)
    end = time.time()
    E3 = np.linalg.norm(np.dot(Aw,X3)-np.ravel(B))
    Erreur_ChoLinalg.append(E3)
    Temps_ChoLinalg.append(end - start)

    start = time.time()
    X4 = np.linalg.solve(Aw,B)
    end = time.time()
    E4 = np.linalg.norm(np.dot(Aw,X4)-(B))
    Erreur_Linalg.append(E4)
    Temps_Linalg.append(end - start)

plt.plot(test, Temps_LU, label ="LU")
plt.plot(test, Temps_Cho, label ="Cholesky")
plt.plot(test, Temps_ChoLinalg, label ="np.linalg.cholesky")
plt.plot(test, Temps_Linalg, label = "np.linalg.solve")

"""plt.semilogy(test, Temps_Gauss, label = "Gauss")
plt.semilogy(test, Temps_LU, label ="LU")
plt.semilogy(test, Temps_PP, label ="PP")
plt.semilogy(test, Temps_PT, label ="PT")
"""

plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul (s)")
plt.legend()
plt.show()

plt.plot(test, Erreur_LU, label ="LU")
plt.plot(test, Erreur_Cho, label ="Cholesky")
plt.plot(test, Erreur_ChoLinalg, label ="np.linalg.cholesky")
plt.plot(test, Erreur_Linalg, label ="np.linalg.solve")

"""plt.semilogy(test, Erreur_Gauss, label = "Gauss")
plt.semilogy(test, Erreur_LU, label ="LU")
plt.semilogy(test, Erreur_PP, label ="PP")
plt.semilogy(test, Erreur_PT, label ="PT")
"""
plt.title("Erreurs de calcul en fonction de la taille de la matrice", fontsize = 10)
plt.xlabel("Taille de la matrice")
plt.ylabel("||AX - B||")
plt.legend()
plt.show()

plt.plot(test, Erreur_LU, label ="LU")
plt.plot(test, Erreur_Cho, label ="Cholesky")

plt.title("Erreurs de calcul en fonction de la taille de la matrice", fontsize = 10)
plt.xlabel("Taille de la matrice")
plt.ylabel("||AX - B||")
plt.legend()
plt.show()