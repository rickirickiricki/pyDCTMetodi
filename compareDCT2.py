import time

from scipy.fftpack import fft, dct, dctn,idct
import numpy as np
import math as m
import random
import PIL.Image
import matplotlib.pyplot as plt

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



#generazione matrici

N = 100
i = 2
timeCustom=[]
timeDCT=[]
matrN=[]


def plotgraph(timeDCT, timeCustom, matrN):


    # corresponding y axis values

    plt.yscale("log")
    plt.plot(matrN,timeDCT,label="DCT")
    plt.plot(matrN,timeCustom,label="Custom")

    # naming the x axis
    plt.xlabel('matrix size')
    # naming the y axis
    plt.ylabel('time')

    # giving a title to my graph
    plt.title('Comparison time')
    plt.legend()
    # function to show the plot
    plt.show()


while i < N:
    array = np.random.randint(0, 255, size=(i, i))
    if i >= 10 and i<100:
        i += 10
    elif i>=100:
        i += 50
    else:
        i += 1
    print(array.shape)
    matrN.append(i)
    #tempi per funzione custom
    start=time.time()
    dct2(array)
    end=time.time()
    start=end-start
    timeCustom.append(start)
    '''print("tempo custom")
    print(start)'''
    #per funzione di default
    start = time.time()
    dct2FromScipy(array)
    end = time.time()
    start = end - start
    timeDCT.append(start)

    '''print("tempo default")
    print(start)'''

plotgraph(timeDCT,timeCustom,matrN)
print(timeCustom)
print(timeDCT)


'''print("predefinita")
print(dct2FromScipy(array))
print("custom")
print(dct2(array))'''










#------prove aggiuntive------------
'''print("rebuilt from library")
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
'''




































