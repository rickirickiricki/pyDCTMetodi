import math as m
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import dct


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


def dct2FromCode(matA):
    n = len(matA)
    output = np.zeros(n)
    for k in range(0, n):
        tmp = 0
        for i in range(0, n):
            tmp += matA[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        if k == 0:
            alpha = m.sqrt(1 / (n))
        else:
            alpha = m.sqrt(2 / (n))
        output[k] = alpha * tmp
    return output


def myDct2(matrice):
    N = matrice.shape[0]
    M = matrice.shape[1]
    matOutput = np.empty([N, M])
    for j in range(M):
        matOutput[:, j] = dct2FromCode(matrice[:, j])
    for i in range(N):
        matOutput[i, :] = dct2FromCode(matOutput[i, :])
    return matOutput


def plotgraph(timeDCT, timeCustom, matrN):
    # corresponding y axis values

    plt.yscale("log")
    plt.plot(matrN, timeDCT, label="DCT")
    plt.plot(matrN, timeCustom, label="Custom")

    # naming the x axis
    plt.xlabel('matrix size')
    # naming the y axis
    plt.ylabel('time')

    # giving a title to my graph
    plt.title('Comparison time')
    plt.legend()
    # function to show the plot
    plt.show()


def createMatrix(i, N):
    timeCustom = []
    timeDCT = []
    matrN = []
    while i < N:
        array = np.random.randint(0, 255, size=(i, i))
        if i >= 10 and i < 100:
            i += 10
        elif i >= 100:
            i += 50
        else:
            i += 1
        matrN.append(i)

        # tempi per funzione custom
        start = time.time()
        myDct2(array)
        end = time.time()
        start = end - start
        timeCustom.append(start)

        # tempi per funzione di default
        start = time.time()
        dct2FromScipy(array)
        end = time.time()
        start = end - start
        timeDCT.append(start)

    plotgraph(timeDCT, timeCustom, matrN)
    df = pd.DataFrame(columns=['Matrix Dimension', 'Time Custom', 'Time Default'])
    df['Matrix Dimension'] = matrN
    df['Time Custom'] = timeCustom
    df['Time Default'] = timeDCT
    df.to_csv("results.csv", index=False)
    print(df)
