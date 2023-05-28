import sys
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
    '''print("dentro delete fre prima")
    print(blocco)'''
    #newArr=[]
    countRiga=0
    countColonna=0
    for riga in blocco:
        for cella in riga:
            if countColonna+countRiga>=d:
                #print(blocco[countRiga][countColonna])
                blocco[countRiga][countColonna]=0 #da controllare in quanto 0 è BIANCO
                #newArr.append(cella)
            countColonna=countColonna+1
        countColonna=0
        countRiga=countRiga+1

    '''print("dentro delete fre dopo")
    print(blocco)'''
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
    print(img)
    img=img.transpose(Image.ROTATE_90)
    a = np.asarray(img)
    print("immagine")
    print(a.shape)

    # arrayGenerale = []
    blocks = blockshaped(a, f)
    height, width = a.shape
    num_blocks_height = height // f
    num_blocks_width = width // f
    new_image_array = np.empty((height, width), dtype=np.uint8)
    for i, block in enumerate(blocks):
        # per ogni blocco dct2
        newBlock = dct2(block)
        '''print("blocco originale")
        print(block)
        print("dct")
        print(newBlock)'''

        # elimino la frequenza in eccesso
        newArr = deleteFrequencies(newBlock, d)
        '''print("no freq")
        print(newArr)'''
        # ricostruisco la idct2 arrotondando i valori
        idctArr = idct2(newArr)
        '''print("IDCT2")
        print(idctArr)
        print(type(idctArr))'''
        block_row = i // num_blocks_width
        block_col = i % num_blocks_width
        start_row = block_row * f
        end_row = start_row + f
        start_col = block_col * f
        end_col = start_col + f
        new_image_array[start_row:end_row, start_col:end_col] = idctArr





    new_image = PIL.Image.fromarray(new_image_array)
    new_image = new_image.transpose(Image.ROTATE_270)
    new_image.save('./imagesExported/'+immagine[9:-4]+'_F'+str(f)+'_D'+str(d)+'.jpg')
    return new_image

#pseudocodice("./images/20x20.bmp",10,5)
#pseudocodice("./images/cathedral.bmp",100,5)
#pseudocodice("./images/20x20.bmp",8,8)
#pseudocodice("./images/640x640.bmp",4,4)