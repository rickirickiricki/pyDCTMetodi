import time

import PIL.Image
from scipy.fftpack import fft, dct, idct
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from skimage.io import imread
from skimage.color import rgb2gray
import os
import cv2
import math as m
import matplotlib.image
from matplotlib import pyplot as plt
def dct2(a):
    size1 = a.shape[0]
    size2 = a.shape[1]
    output = np.empty([size1, size2])

    # DCT2 (DCT by row and then by column)
    for i in range(0, size1):
        output[i] = dct(a[i], 2, norm='ortho')

    for i in range(0, size2):
        output[:, i] = dct(output[:, i], 2, norm='ortho')
    return output
def idct2Custom(a):
    size1 = a.shape[0]
    size2 = a.shape[1]
    output = np.empty([size1, size2])

    # DCT2 (DCT by row and then by column)
    for i in range(0, size1):
        output[i] = idct(a[i], 2, norm='ortho')

    for i in range(0, size2):
        output[:, i] = idct(output[:, i], 2, norm='ortho')
    return output

def idct2(blocco):
    #ris=dct(dct(blocco.T, norm='ortho').T, norm='ortho')
    ris = idct2Custom(blocco)
    countRiga = 0
    countColonna = 0
    for riga in ris:
        for cella in riga:
            if(cella<0):
                ris[countRiga][countColonna]=0
            elif(cella>255):
                ris[countRiga][countColonna]=255
            countColonna=countColonna+1
        countColonna = 0
        countRiga = countRiga + 1
    return ris

def deleteFrequencies(blocco,d):
    #newArr=[]
    countRiga=0
    countColonna=0
    for riga in blocco:
        for cella in riga:
            if countColonna+countRiga>=d:
                #print(blocco[countRiga][countColonna])
                blocco[countRiga][countColonna]=0
                #newArr.append(cella)
            countColonna=countColonna+1
        countColonna=0
        countRiga=countRiga+1
    #print(blocco)
    return blocco

def blockshaped(arrayImg, F):
    height, width = arrayImg.shape
    blocks = []
    for y in range(0, height // F * F, F):
        for x in range(0, width // F * F, F):
            block = arrayImg[y:y + F, x:x + F]
            blocks.append(block)
    '''for i, block in enumerate(blocks):
        print(block)'''

    return blocks


def pseudocodice(immagine,f,d):
    # divido l'immagine in F x F
    img = Image.open(immagine)
    #img=img.transpose(Image.ROTATE_90)
    a = np.asarray(img)
    print("immagine")
    print(a.shape)

    arrayGenerale=[]
    nBlocchi=len(blockshaped(a,f))
    for block in blockshaped(a,f):

        #per ogni blocco dct2
        newBlock=dct2(block)
        '''print("blocco originale")
        print(block)
        print("dct")
        print(newBlock)'''

        #elimino la frequenza in eccesso
        newArr=deleteFrequencies(newBlock,d)
        '''print("no freq")
        print(newArr)'''
        #ricostruisco la idct2 arrotondando i valori
        idctArr=idct2(newArr)
        '''print("IDCT2")
        print(idctArr)
        print(type(idctArr))'''


        arrayGenerale.append(idctArr)


    #ricompongo immagine mettendo insieme blocchi nell'ordine giusto
    array = np.array(arrayGenerale, dtype=np.uint8)
    n=int(m.sqrt(nBlocchi))
    print("\n\nCon sta roba concateno le varie colonne una accanto all'altra:\n",
          [np.column_stack(array[start:start + n]) for start in range(0, array.shape[0] - n + 1, n)])

    block_fin = np.concatenate(
        [np.column_stack(array[start:start + n]) for start in range(0, array.shape[0] - n + 1, n)])
    print("\n\nblocco finale: \n", block_fin)

    new_image = PIL.Image.fromarray(block_fin)
    new_image.save('./imagesExported/'+immagine[9:-4]+'_F'+str(f)+'_D'+str(d)+'.png')


#pseudocodice("./images/20x20.bmp",10,5)
pseudocodice("./images/cathedral.bmp",10,18)
#pseudocodice("./images/20x20.bmp",8,8)
#pseudocodice("./images/640x640.bmp",4,4)