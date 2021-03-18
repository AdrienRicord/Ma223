"""Programme TP1 Ma223 @AdrienRicord @VincentLenoble"""

#Initialisation...

import matplotlib.pyplot as plt
import numpy as np
import time

A = np.random.random(size=(10,10))
B = np.random.random(size=(10,1))

#A = np.array([[1,1,1,1],[2,4,-3,2],[-1,-1,0,-3],[1,-1,4,9]])
#B = np.array([[1],[1],[2],[-8]])

#Fonctions...

def ReductionGauss(Aaug):

    n,m = Aaug.shape

    for i in range(0,n-1):
        for k in range(i+1,n):
            Aaug[k,:] = Aaug[k,:]-(Aaug[k,i]/Aaug[i,i])*Aaug[i,:]

    return Aaug

def ResolutionSystTriSup(Taug):

    T = np.copy(Taug)
    n,m = T.shape
    x = np.zeros(n)

    x[n-1] = T[n-1,n] / T[n-1,n-1]      
    for i in range(n-2,-1,-1):
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

def GaussChoixPivotPartiel(A,B):

    Aaug = np.concatenate((A,B), axis=1)
    n,m = Aaug.shape
    X = np.zeros(n)
    
    for i in range(n):
        for k in range(i+1,n):
            if abs(Aaug[k,i]) > abs(Aaug[i,i]):
                Aw = np.copy(Aaug[i,:])
                Aaug[i,:] = Aaug[k,:]
                Aaug[k,:] = Aw
            p = Aaug[i,i]

        if p != i:
            for k in range(i+1,n):
                Aaug[k,:] = Aaug[k,:]-(Aaug[k,i]/p)*Aaug[i,:]

    X[n-1] = Aaug[n-1,n] / Aaug[n-1,n-1]      
    for i in range(n-2,-1,-1):
        X[i] = (Aaug[i,n] - np.sum(Aaug[i,i+1:n]*X[i+1:n]))/Aaug[i,i]

    return X

def GaussChoixPivotTotal(A,B):

    n,m = np.shape(A)
    
    X = np.zeros(n)
    l = [0,0]
    c = [0,0]

    imax = 0
    jmax = 0

    for k in range(0, n-1):

        imax = k
        jmax = k
        p = abs(A[imax,jmax])

        if k >= 1:
            for j in range(k,n-1):
                for i in range(k,n-1):
                    if abs(A[i,j]) > p : 
                        imax = i
                        jmax = j
                        p = abs(A[imax,jmax])

            c[:] = A[:,k]
            A[:,k] = A[:,jmax]
            A[:,jmax] = c[:]

            l[:] = A[k,:]
            A[k,:] = A[imax,:]
            A[imax,:] = l[:]

        for j in range(k+1,n):
            A[j,:] = A[j,:]-(A[j,k]/A[k,k])*A[k,:]

    Aaug = np.concatenate((A,B), axis = 1)

    X[n-1] = Aaug[n-1,n] / Aaug[n-1,n-1]      
    for i in range(n-2,-1,-1):
        X[i] = (Aaug[i,n] - np.sum(Aaug[i,i+1:n]*X[i+1:n]))/Aaug[i,i]

    return X

#Programme...

print("--------------------Algorithme de Gauss--------------------\n")

X = Gauss(A,B)
print("La matrice solution :\n", X)

e = np.linalg.norm(np.dot(A,X)-np.ravel(B))
print("Erreur Gauss", e)


print("\n--------------------Décomposition LU--------------------\n")

L,U = DecompositionLU(A)
print("La matrice L :\n", L, "\nla matrice U :\n", U)

X = ResolutionLU(L,U,B)
print("La matrice solution :\n", X)

LU = np.matmul(L,U)
e = np.linalg.norm(np.dot(LU,X)-np.ravel(B))
print("Erreur LU", e)


print("\n--------------------Pivot Partiel--------------------\n")

X = GaussChoixPivotPartiel(A,B)
print("La matrice solution: \n", X)

e = np.linalg.norm(np.dot(A,X)-np.ravel(B))
print("Erreur PP", e)


print("\n--------------------Pivot Total----------------------\n")

X = GaussChoixPivotTotal(A,B)
print("La matrice solution: \n", X)

e = np.linalg.norm(np.dot(A,X)-np.ravel(B))
print("Erreur PT", e)

#Matplotlib...

print("\n--------------------Graphiques-----------------------\n")

test = []
for i in range(2,502,50): #de ... à ... avec un pas de ...
    test.append(i)

len_test = len(test)
Erreur_Gauss = []
Erreur_LU = []
Erreur_PP = []
Erreur_PT = []

Temps_Gauss = []
Temps_LU = []
Temps_PP = []
Temps_PT = []

for i in range(len_test):
    
    n = test[i]
    A = np.random.random(size=(n,n))
    B = np.random.random(size=(n,1))
    A1 = np.copy(A)
    A2 = np.copy(A)
    A3 = np.copy(A)
    A4 = np.copy(A)

    start = time.time()
    X = Gauss(A1,B)
    end = time.time()
    E = np.linalg.norm(np.dot(A1,X)-np.ravel(B))
    Erreur_Gauss.append(E)
    Temps_Gauss.append(end - start)

    start = time.time()
    L,U = DecompositionLU(A2)
    X2 = ResolutionLU(L,U,B)
    LU = np.matmul(L,U)
    end = time.time()
    E2 = np.linalg.norm(np.dot(A2,X2)-np.ravel(B))
    Erreur_LU.append(E2)
    Temps_LU.append(end - start)

    start = time.time()
    X3 = GaussChoixPivotPartiel(A3,B)
    end = time.time()
    E3 = np.linalg.norm(np.dot(A3,X3)-np.ravel(B))
    Erreur_PP.append(E3)
    Temps_PP.append(end - start)

    start = time.time()
    X4 = GaussChoixPivotTotal(A4,B)
    end = time.time()
    E4 = np.linalg.norm(np.dot(A4,X4)-np.ravel(B))
    Erreur_PT.append(E4)
    Temps_PT.append(end - start)

plt.plot(test, Temps_Gauss, label = "Gauss")
plt.plot(test, Temps_LU, label ="LU")
plt.plot(test, Temps_PP, label ="PivotPartiel")
plt.plot(test, Temps_PT, label ="PivotTotal")

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

plt.plot(test, Temps_Gauss, label = "Gauss")
plt.plot(test, Temps_LU, label ="LU")
plt.plot(test, Temps_PP, label ="PivotPartiel")

plt.title("Temps de calcul en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Temps de calcul (s)")
plt.legend()
plt.show()

plt.plot(test, Erreur_Gauss, label = "Gauss")
plt.plot(test, Erreur_LU, label ="LU")
plt.plot(test, Erreur_PP, label ="PivotPartiel")
plt.plot(test, Erreur_PT, label ="PivotTotal")

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

plt.plot(test, Erreur_PP, 'g', label ="PivotPartiel")
plt.plot(test, Erreur_PT, 'r', label ="PivotTotal")

plt.title("Erreurs de calcul en fonction de la taille de la matrice", fontsize = 10)
plt.xlabel("Taille de la matrice")
plt.ylabel("||AX - B||")
plt.legend()
plt.show()
