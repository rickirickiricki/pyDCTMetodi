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

    return output


def dct2FromCode(V):

    n = len(V)
    output = np.zeros(n)

    for k in range(0, n):
        tmp=0
        for i in range(0, n):
            tmp += V[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        if k == 0:
            alpha = m.sqrt(1 / (n))
        else:
            alpha = m.sqrt(2/ (n))
        output[k] = alpha * tmp
    return output
def dct2(matrice):
    #N, M = matrice.shape
    N=matrice.shape[0]
    M=matrice.shape[1]
    matOutput = np.empty([N, M])
    #C = np.zeros((N, M), dtype='float')
    # sommatoria su N
    for j in range(M):
        matOutput[:, j] = dct2FromCode(matrice[:, j])

    for i in range(N):
        matOutput[i, :] = dct2FromCode(matOutput[i, :])



    return matOutput

print("predefinita")
print(dct2FromScipy(array))
print("custom")
print(dct2(array))


#------prove aggiuntive------------
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




































