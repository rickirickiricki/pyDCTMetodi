import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2 as cv

from comprimeIMG import solve
from PIL import Image


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

    def calculate(self):
        self.pathImage = str(self.pathImage).split("/")[-1]
        self.pathImage = "./images/" + self.pathImage
        print("prima di chiamare pseudo " + self.pathImage)
        array = solve(self.pathImage, self.F, self.D)
        print("termina pseudo")
        print(array)
        self.finalImage.setPixmap(QPixmap(array).scaled(int(round(self.width / 1.8)), int(round(self.height / 1.8)), Qt.KeepAspectRatio))


if __name__ == '__main__':
    applicationGUI = QApplication(sys.argv)
    ex = createGUI()
    sys.exit(applicationGUI.exec_())