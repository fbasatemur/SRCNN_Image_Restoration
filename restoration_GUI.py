# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:49:20 2020

@author: fbasatemur
"""

from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QAction, QFileDialog
from PyQt5.QtGui import  QPixmap, QFont
from PyQt5.QtCore import Qt

from data_preprocessing import predict, convert_cv_qt
from SRCNN_model import model
from keras import backend


class Window(QMainWindow):
      
      def __init__(self):
            super().__init__()
            
            # main window
            self.width = 950
            self.height = 500
            
            self.setWindowTitle("SRCNN Restoration")
            self.setGeometry(200, 200, self.width, self.height)
#            self.setWindowIcon(QIcon("icon.jpg"))           
                         
            self.showMenubar()
            
            self.labels()   
            
            self.showSaveButton()
            self.showApplyButton()
            
            self.inputImageShow()
            self.outputImageShow()
            
            self.model = model()
            
            self.show()
            
            
      def __del__(self):
            del self.model 
            backend.clear_session()
            
            
      def showMenubar(self):
            menu = self.menuBar()
            fileMenu = menu.addMenu('File')
            
            openAction = QAction('Open', self)  
            fileMenu.addAction(openAction)
            openAction.triggered.connect(self.openImage) 
            
            
      def openImage(self):
            self.imagePath, _ = QFileDialog.getOpenFileName(self, filter="Image Files (*.jpg *.bmp *.png)")
            self.setInputPixmap(QPixmap(self.imagePath))
            
            
      def labels(self):
            self.labelInputText = QLabel("Input ", self)
            self.labelInputText.move(50,40)
            self.labelInputText.setFont(QFont("Arial", 9, QFont.Bold))
            
            self.labelOutputText = QLabel("SRCNN ", self)
            self.labelOutputText.move(500,40)
            self.labelOutputText.setFont(QFont("Arial", 9, QFont.Bold))
                              
            
      def inputImageShow(self):
            self.inputImage = QLabel(self)
            self.inputImage.resize(400, 400)
            self.inputImage.move(50, 80)
            
            
      def outputImageShow(self):
            self.outputImage = QLabel(self)
            self.outputImage.resize(400, 400)
            self.outputImage.move(500, 80)


      def showApplyButton(self):
            self.applyBtn = QPushButton("Apply", self)
            self.applyBtn.move(680,40)
            
            self.applyBtn.clicked.connect(self.applyBtnFunction)
            
            
      def showSaveButton(self):
            self.saveBtn = QPushButton("Save", self)
            self.saveBtn.move(800,40)
            
            self.saveBtn.clicked.connect(self.saveBtnFunction)
      
          
      def setInputPixmap(self, pixmap):   
            smaller_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.inputImage.setPixmap(smaller_pixmap)
            
            
      def setOutputPixmap(self, qImage):
            self.qImage_scaled = qImage.scaled(400, 400, Qt.KeepAspectRatio)
            self.outputImage.setPixmap(QPixmap.fromImage(self.qImage_scaled))

      
      def applyBtnFunction(self):
            ret_model = self.model.get_model()
            srcnn_BGR = predict(self.imagePath, ret_model)
            self.srcnn_qt = convert_cv_qt(srcnn_BGR)
            self.setOutputPixmap(self.srcnn_qt)
            
            
      def saveBtnFunction(self):
            self.srcnn_qt.save(self.imagePath.split("/")[-1], format="bmp")


      