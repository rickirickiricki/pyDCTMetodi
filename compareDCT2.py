from scipy.fftpack import fft, dct, dctn,idct
import numpy as np
import math as m
import PIL.Image
'''array=np.array([[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,
  255,255],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0],
 [255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0]])'''
array=np.array([[231,    32,   233,   161,    24,    71,   140,   245],
   [247 ,   40   ,248,   245   ,124,   204   , 36,   107],
   [234 ,  202   ,245 ,  167   ,  9 ,  217   ,239 ,  173],
   [193  , 190   ,100  , 167   , 43  , 180  ,   8  ,  70],
    [11   , 24  , 210  , 177   , 81  , 243   ,  8   ,112],
    [97  , 195   ,203  ,  47  , 125  , 114  , 165   ,181],
   [193   , 70 ,  174   ,167 ,   41  ,  30  , 127   ,245],
    [87   ,149,    57   ,192,    65  , 129 ,  178   ,228]])
def dct2FromScipy(a):
    size1 = a.shape[0]
    size2 = a.shape[1]
    output = np.empty([size1, size2])

    # DCT2 (DCT by row and then by column)
    for i in range(0, size1):
        output[i] = dct(a[i], 2, norm='ortho')

    for i in range(0, size2):
        output[:, i] = dct(output[:, i], 2, norm='ortho')
    ris=dct(a,2)
    #ris = dct(dct(a.T, norm='ortho').T, norm='ortho')

    print("V1")
    #print(dctn(a, norm='ortho'))
    return output


def dct2FromCode(V):
    """
    Implementation from pdf formula (Parte_2.pdf).
    """
    '''N = len(V)
    c = np.zeros(N)
    for k in range(N):
        #s = 0
        for i in range(N):
            c[k] += V[i] * np.cos(k * np.pi * ((2*i+1) / (2*N)))
        #c[k]*=2
        if k == 0:
            alpha = 1 / m.sqrt(N)
        else:
            alpha=m.sqrt(2 / N)

        c[k] =c[k]* alpha'''
    #           N-1
    # y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
    #           n=0
    n = len(V)
    output = np.zeros(n)

    for k in range(0, n):
        s=0
        for i in range(0, n):
            s += V[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))

        # If norm='ortho', y[k] is multiplied by a scaling factor f:
        #  f = sqrt(1/(4*N)) if k = 0,
        #  f = sqrt(1/(2*N)) otherwise.
        '''if k == 0:
            output[k] *= m.sqrt(1 / (4 * n))
        else:
            output[k] *= m.sqrt(1 / (2 * n))'''
        if k == 0:
            alpha = m.sqrt(1 / (n))
        else:
            alpha = m.sqrt(2/ (n))
        output[k] = alpha * s
    return output
def dct2(matrice):
    #N, M = matrice.shape
    N=matrice.shape[0]
    M=matrice.shape[1]
    C = np.empty([N, M])
    #C = np.zeros((N, M), dtype='float')
    # sommatoria su N
    for j in range(M):
        C[:, j] = dct(matrice[:, j])

    for i in range(N):
        C[i, :] = dct(C[i, :])
    '''for i in range(0,N):
        C[i] = dct2FromCode(matrice[i])

    #sommatoria su M
    for i in range(0,M):
        C[:, i] = dct2FromCode(C[:, i])'''



    return C

print("predefinita")
print(dct2FromScipy(array))
print("custom")
print(dct2(array))
print("rebuilt from library")
print(idct(dct2FromScipy(array),2))
new_image = PIL.Image.fromarray((idct(dct2FromScipy(array),2)))
print(new_image)
new_image = new_image.convert('RGB')
new_image.save('PREDEFINITA.png')

print("rebuilt from custom")
print(idct(dct2(array),2))

new_image = PIL.Image.fromarray(idct(dct2(array),2))
new_image = new_image.convert('RGB')
new_image.save('CUSTOM.png')




































