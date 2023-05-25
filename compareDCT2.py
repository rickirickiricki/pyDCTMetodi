from scipy.fftpack import fft, dct
import numpy as np
import math as m
array=np.array([[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]])
def dct2FromScipy(a):
    ris=dct(a,2)
    print(ris)
    return ris
def dct2FromCode(a):
    M= len(a.shape[1])
    N=len(a.shape[0])
    matB=[]
    for riga in a:

        for colonna in a[riga]:
            ris = 1 / (riga * riga)
            for i in range(N-1):
               ris1=ris*(1/(colonna*colonna))
               for j in range(M-1):
                   ris2=ris1*(a[i][j]*m.cos((colonna*3.14)*(2j+1)/(2*M)))
               ris3=ris2*m.cos((riga*3.14)*(2*i+1)/(2*N))
            matB[riga][colonna]=ris3
    return matB
dct2FromScipy(array)