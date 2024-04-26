# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ice_image.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

import preprocessor
from preprocessor import image_preprocessor
from model_handler import model_handler
from image_convertor import image_convertor
from PIL import Image

import sys
from os import getcwd


class QLabelClickable(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(QLabelClickable, self).__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()


class Ui_Dialog(QtWidgets.QDialog):
    path = ""

    def __init__(self, parent=None):
        super(Ui_Dialog, self).__init__(parent)

        self.setWindowTitle("Определение сплоченности льда")
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(1096, 600)
        self.path = ""

        self.initUI()

    def initUI(self):
        self.add_image = QtWidgets.QPushButton("Загрузить изображение", self)
        self.add_image.setGeometry(QtCore.QRect(24, 24, 164, 24))
        self.add_image.setObjectName("add_image")
        self.add_image.clicked.connect(self.seleccionarImagen)

        self.create_mask = QtWidgets.QPushButton("Создать маску", self)
        self.create_mask.setGeometry(QtCore.QRect(372, 24, 164, 24))
        self.create_mask.setObjectName("create_mask")
        self.create_mask.clicked.connect(self.create_mask_m)

        self.image = QtWidgets.QLabel(self)
        self.image.setGeometry(QtCore.QRect(24, 64, 512, 512))
        self.image.setObjectName("image")

        self.mask = QtWidgets.QLabel(self)
        self.mask.setGeometry(QtCore.QRect(560, 64, 512, 512))
        self.mask.setObjectName("mask")

    def seleccionarImagen(self):
        imagen, extension = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                  "Выбор картинки",
                                                                  "D:\Dowlands\ICES_P\READY_IMAGES\images",
                                                                  "Image (*.png *.jpg)",
                                                                  options=QtWidgets.QFileDialog.Options())

        if imagen:
            pixmapImagen = QtGui.QPixmap(imagen).scaled(512,
                                                        512,
                                                        QtCore.Qt.KeepAspectRatio,
                                                        QtCore.Qt.SmoothTransformation)
            self.image.setPixmap(pixmapImagen)
            self.path = imagen

    def create_mask_m(self):
        ip = image_preprocessor()
        image = ip.preprocess(self.path)

        mh = model_handler()
        pr_mask = mh.predict(image)

        ic = image_convertor()
        mask = ic.mask_to_image(pr_mask)
        mask.save('mask.png')

        pixmapImagen = QtGui.QPixmap("mask.png").scaled(512,
                                                       512,
                                                       QtCore.Qt.KeepAspectRatio,
                                                       QtCore.Qt.SmoothTransformation)
        self.mask.setPixmap(pixmapImagen)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = Ui_Dialog()
    main.show()
    sys.exit(app.exec_())