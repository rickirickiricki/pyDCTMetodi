import pandas as pd
from PIL import Image
import numpy as np


def comprime(imgPath, F, d):
    if d < 0 | d > (2 * F - 2):
        print("d non valida")
    img = Image.open(imgPath)
    arrayImg = np.asarray(img)
    #df = pd.DataFrame(arrayImg)
    #excel_file = 'array_data1.xlsx'  # Specify the desired file name
    #df.to_excel(excel_file, index=False)
    height, width = arrayImg.shape
    blocks = []
    for y in range(0, height // F * F, F):
        for x in range(0, width // F * F, F):
            block = arrayImg[y:y + F, x:x + F]
            blocks.append(block)
    for i, block in enumerate(blocks):
        print(block)


F = 8
d = 10
comprime("./images/cathedral.bmp", F, d)
