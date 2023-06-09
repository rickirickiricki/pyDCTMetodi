import PIL.Image
from scipy.fftpack import fft, dct, idct
import numpy as np
from PIL import Image

def dct2(a):
    size1 = a.shape[0]
    size2 = a.shape[1]
    output = np.empty([size1, size2])
    for i in range(0, size1):
        output[i] = dct(a[i], 2, norm='ortho')
    for i in range(0, size2):
        output[:, i] = dct(output[:, i], 2, norm='ortho')
    return output

def idct2Custom(a):
    size1 = a.shape[0]
    size2 = a.shape[1]
    output = np.empty([size1, size2])
    for i in range(0, size1):
        output[i] = idct(a[i], 2, norm='ortho')
    for i in range(0, size2):
        output[:, i] = idct(output[:, i], 2, norm='ortho')
    return output

def idct2(blocco):
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
    countRiga=0
    countColonna=0
    for riga in blocco:
        for cella in riga:
            if countColonna+countRiga>=d:
                blocco[countRiga][countColonna]=0
            countColonna=countColonna+1
        countColonna=0
        countRiga=countRiga+1
    return blocco

def blockshaped(arrayImg, F):
    height, width = arrayImg.shape
    blocks = []
    for y in range(0, height // F * F, F):
        for x in range(0, width // F * F, F):
            block = arrayImg[y:y + F, x:x + F]
            blocks.append(block)
    return blocks

def solve(immagine, f, d):

    # divido l'immagine in F x F
    img = Image.open(immagine).convert("L")
    img=img.transpose(Image.ROTATE_90)
    a = np.asarray(img)

    blocks = blockshaped(a,f)
    height, width = a.shape
    num_blocks_height = height // f
    new_height = height - (height % f)

    num_blocks_width = width // f
    new_width = width - (width % f)
    new_image_array = np.empty((new_height, new_width), dtype=np.uint8)
    for i, block in enumerate(blocks):
        # per ogni blocco dct2
        newBlock = dct2(block)

        # elimino la frequenza in eccesso
        newArr = deleteFrequencies(newBlock, d)

        # ricostruisco la idct2 arrotondando i valori
        idctArr = idct2(newArr)
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
    pathLocal='./imagesExported/'+immagine[9:-4]+'_F'+str(f)+'_D'+str(d)+'.jpg'
    return pathLocal

#solve("./images/20x20.bmp", 10, 5)