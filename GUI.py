import numpy as np
import pandas as pd
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pathlib
import cv2 as cv
import PIL.Image
from comprimeIMG import pseudocodice, blockshaped, dct2, deleteFrequencies, idct2Custom, idct2
from PIL import Image
from scipy.fftpack import fft, dct, idct
from skimage import io
from skimage.io import imread
from skimage.color import rgb2gray

class createGUI(QWidget):
    F = 1
    D = 0
    limitF = 0
    limitD = 0
    pathImage = "default"

    def __init__(self):  # constructor initializes the object's attributes
        super().__init__()
        self.title = "DCT program interface"
        screen_resolution = applicationGUI.desktop().screenGeometry()
        self.width = screen_resolution.width()
        self.height = screen_resolution.height()
        self.x = 100
        self.y = 100
        self.originalImage = QLabel(self)
        self.finalImage = QLabel(self)
        self.initializeGUI()

    def initializeGUI(self):
        grid = QGridLayout()  # create a grid for widgets
        grid.addWidget(self.selectUpload(), 0, 0, 1, 2)  # button upload
        grid.addWidget(self.createOriginalImage(), 1, 0)  # load original image
        grid.addWidget(self.createFinalImage(), 1, 1)  # load final image
        self.setLayout(grid)  # after setting widget to the grid that code create the grid layout
        self.setWindowTitle(self.title)  # window title
        self.setGeometry(self.x, self.y, 500, 240)
        self.show()

    def selectUpload(self):
        widget = QGroupBox('Upload your Image and select parameters')
        button = QPushButton('Upload', self)
        button.setFixedSize(80,30)
        button.clicked.connect(self.getImage)
        vbox = QVBoxLayout()
        vbox.addWidget(button, alignment=Qt.AlignHCenter)
        self.value_f = QLabel('Select the block dimension F')
        self.spinboxf = QSpinBox()
        self.spinboxf.setMinimum(1)
        self.spinboxf.setMaximum(10000)
        self.spinboxf.valueChanged.connect(self.controlValues)
        self.value_d = QLabel('Select frequency elimination D')
        self.spinboxd = QSpinBox()
        self.spinboxd.setMinimum(0)
        self.spinboxd.setMaximum(10000)
        self.spinboxd.valueChanged.connect(self.controlValues)
        vbox.addStretch(1)
        vbox.addWidget(self.value_f)
        vbox.addWidget(self.spinboxf)
        vbox.addWidget(self.value_d)
        vbox.addWidget(self.spinboxd)
        vbox.addStretch(1)
        button2 = QPushButton('Calculate', self)
        #button2.clicked.connect(lambda pseudocodice: pseudocodice(self.pathImage, self.F, self.D))
        button2.clicked.connect(self.calculate)
        button2.setFixedSize(80, 30)
        vbox.addWidget(button2, alignment=Qt.AlignHCenter)
        vbox.addStretch(1)
        vbox.setAlignment(Qt.AlignCenter)
        widget.setLayout(vbox)
        return widget



    def createOriginalImage(self):
        widget = QGroupBox('Original Image')
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.originalImage)
        widget.setLayout(vbox)
        return widget

    def controlValues(self):  # checks for correctness of the values
        self.F = self.spinboxf.value()
        self.D = self.spinboxd.value()
        self.limitD = (2 * self.F) - 2
        self.spinboxd.setMaximum(self.limitD)

    def createFinalImage(self):
        widget = QGroupBox('Processed Image')
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.finalImage)
        widget.setLayout(vbox)
        return widget

    def getImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', "Image Files (*.bmp)")
        self.pathImage = fileName
        grayScaleCheck = QPixmap(fileName).toImage().isGrayscale()
        if grayScaleCheck == False:
            message = QMessageBox()
            message.setIcon(QMessageBox.Critical)
            message.setInformativeText('Not Grayscale Image')
            message.setWindowTitle("Error")
            message.exec_()
        else:
            imageRead = cv.imread(fileName)
            self.imageHeight = imageRead.shape[0]
            self.imageWidth = imageRead.shape[1]
            self.originalImage.setPixmap(QPixmap(fileName).scaled(int(round(self.width / 1.8)), int(round(self.height / 1.8)),Qt.KeepAspectRatio))
            self.limitF = min(self.imageHeight, self.imageWidth)
            self.spinboxf.setMaximum(self.limitF)

        img = Image.open(fileName)
        print(img.filename)
        return img.filename

    def calculate(self):  # richiama pseudocodice per calcolare l immagine,scriverla nella catella images come tmp.bmp e settare finalImage=tmp
        print("prima di chiamare pseudo " + self.pathImage)
        array = pseudocodice(self.pathImage, self.F, self.D)
        print("termina pseudo")
        tmp = str(pathlib.Path(__file__).parent.absolute()) + "/temp.bmp"
        cv.imwrite(tmp, array)
        self.finalImage.setPixmap(QPixmap(tmp).scaled(int(round(self.width / 1.8)), int(round(self.height / 1.8)), Qt.KeepAspectRatio))


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
    # ris=dct(dct(blocco.T, norm='ortho').T, norm='ortho')
    ris = idct2Custom(blocco)
    countRiga = 0
    countColonna = 0
    for riga in ris:
        for cella in riga:
            if (cella < 0):
                ris[countRiga][countColonna] = 0
            elif (cella > 255):
                ris[countRiga][countColonna] = 255
            countColonna = countColonna + 1
        countColonna = 0
        countRiga = countRiga + 1
    return ris


def deleteFrequencies(blocco, d):
    '''print("dentro delete fre prima")
    print(blocco)'''
    # newArr=[]
    countRiga = 0
    countColonna = 0
    for riga in blocco:
        for cella in riga:
            if countColonna + countRiga >= d:
                # print(blocco[countRiga][countColonna])
                blocco[countRiga][countColonna] = 0  # da controllare in quanto 0 è BIANCO
                # newArr.append(cella)
            countColonna = countColonna + 1
        countColonna = 0
        countRiga = countRiga + 1

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


def pseudocodice(immagine, f, d):
    print(f)
    print(d)
    print("dentro pseudo " + immagine)
    # divido l'immagine in F x F
    img = Image.open(immagine).convert("L")
    img = img.transpose(Image.ROTATE_90)
    a = np.asarray(img)
    print("immagine")
    print(a.shape)
    # arrayGenerale = []

    blocks = blockshaped(a, f)
    height, width = a.shape
    num_blocks_height = height // f
    new_height = height - (height % f)

    num_blocks_width = width // f
    new_width = width - (width % f)
    new_image_array = np.empty((new_height, new_width), dtype=np.uint8)
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
    new_image.save('./imagesExported/' + immagine[9:-4] + '_F' + str(f) + '_D' + str(d) + '.jpg')


# pseudocodice("./images/20x20.bmp",10,5)
# pseudocodice("./images/cathedral.bmp",100,198)
# pseudocodice("./images/20x20.bmp",8,8)
# pseudocodice("./images/640x640.bmp",4,4)


if __name__ == '__main__':
    applicationGUI = QApplication(sys.argv)
    ex = createGUI()
    sys.exit(applicationGUI.exec_())