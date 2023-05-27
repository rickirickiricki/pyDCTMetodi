import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pathlib
import cv2 as cv

class createGUI(QWidget):
    F = 0
    D = 0
    limitF = 0
    limitD = 0
    pathImage = "default_path"

    def __init__(self): #constructor initializes the object's attributes
        super().__init__()
        self.title = "DTC program interface"
        screen_resolution = applicationGUI.desktop().screenGeometry()
        self.width = screen_resolution.width()
        self.height = screen_resolution.height()
        self.x = 100
        self.y = 100
        self.originalImage = QLabel(self)
        self.finalImage = QLabel(self)
        self.initializeUI()


    def initializeUI(self):
        grid = QGridLayout() #create a grid for widgets
        grid.addWidget(self.selectUpload(), 0, 0) #button open
        grid.addWidget(self.selectParameters(), 0, 1) #user parameters
        grid.addWidget(self.createOriginalImage(), 1, 0) #load original image
        grid.addWidget(self.createFinalImage(), 1, 1) #load final image
        self.setLayout(grid) #after setting widget to the grid that code create the grid layout
        self.setWindowTitle(self.title) #window title
        self.setGeometry(self.x, self.y, int(round(self.width / 2)), int(round(self.height / 2)))
        self.show()


    def selectUpload(self):
        widget = QGroupBox('Upload your Image')
        button = QPushButton('Upload', self)
        button.clicked.connect(self.getImage)
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(button)
        widget.setLayout(vbox)
        return widget

    def selectParameters(self):
        widget = QGroupBox('Parameters for compression')
        self.f = QLabel('F')
        self.spinboxf = QSpinBox()
        self.spinboxf.setMinimum(1)
        self.spinboxf.valueChanged.connect(self.value_changed)
        self.d = QLabel('d')
        self.spinboxd = QSpinBox()
        self.spinboxd.setMinimum(0)
        self.spinboxd.valueChanged.connect(self.value_changed)
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.f)
        vbox.addWidget(self.spinboxf)
        vbox.addWidget(self.d)
        vbox.addWidget(self.spinboxd)
        vbox.addStretch(1)
        button2 = QPushButton('Calculate', self)
        #button2.clicked.connect(self.elaboraImmagine/comprimiImmagine) #####collegare a metodo di compressione/elaborazione
        vbox.addWidget(button2)
        vbox.addStretch(1)
        widget.setLayout(vbox)
        return widget


    def createOriginalImage(self):
        widget = QGroupBox('Original Image')
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.originalImage)
        widget.setLayout(vbox)
        return widget

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
            message.setText("Error")
            message.setInformativeText('This File is not a grayscale Image')
            message.setWindowTitle("Error")
            message.exec_()
        else:
            imageRead = cv.imread(fileName)
            self.imageHeight = imageRead.shape[0]
            self.imageWidth = imageRead.shape[1]
            imageChannel = imageRead.shape[2]
            self.originalImage.setPixmap(QPixmap(fileName).scaled(int(round(self.width / 1.8)), int(round(self.height / 1.8)), Qt.KeepAspectRatio))
            self.limitF = min(self.imageHeight, self.imageWidth)
            self.spinboxf.setMaximum(self.limitF)


    def value_changed(self):
        self.D = self.spinboxd.value()
        self.F = self.spinboxf.value()
        self.limitD = (2 * self.value_F) - 2
        self.spinboxd.setMaximum(self.limitD)


if __name__ == '__main__':
    applicationGUI = QApplication(sys.argv)
    ex = createGUI()
    sys.exit(applicationGUI.exec_())