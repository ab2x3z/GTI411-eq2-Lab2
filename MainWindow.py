# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


#from Lab1.Lab1_Window import Ui_Lab1_Window
from PyQt5 import QtCore, QtGui, QtWidgets
from Lab2.Lab2_Window import Ui_Lab2_Window
#from Lab3_Window import Ui_Lab3_Window
#from Lab4_Window import Ui_Lab4_Window

class Ui_MainWindow(object):

    #def open_Lab1_Window(self):
        #self.window = QtWidgets.QMainWindow()
        #self.ui = Ui_Lab1_Window()
        #self.ui.setupUi(self.window)
        #MainWindow.hide()
        #self.window.show()

    def open_Lab2_Window(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Lab2_Window()
        self.ui.setupUi(self.window)
        #MainWindow.hide()
        self.window.show()

    #def open_Lab3_Window(self):
        #self.window = QtWidgets.QMainWindow()
        #self.ui = Ui_Lab3_Window()
        #self.ui.setupUi(self.window)
        ##MainWindow.hide()
        #self.window.show()

    #def open_Lab4_Window(self):
        #self.window = QtWidgets.QMainWindow()
        #self.ui = Ui_Lab4_Window()
        #self.ui.setupUi(self.window)
        ##MainWindow.hide()
        #self.window.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.lab1_button = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lab1_button.sizePolicy().hasHeightForWidth())
        self.lab1_button.setSizePolicy(sizePolicy)
        self.lab1_button.setObjectName("lab1_button")
        self.gridLayout.addWidget(self.lab1_button, 0, 0, 1, 1)
        self.lab2_button = QtWidgets.QPushButton(self.centralwidget)
        self.lab2_button.setObjectName("lab2_button")
        self.gridLayout.addWidget(self.lab2_button, 1, 0, 1, 1)
        self.lab3_button = QtWidgets.QPushButton(self.centralwidget)
        self.lab3_button.setObjectName("lab3_button")
        self.gridLayout.addWidget(self.lab3_button, 2, 0, 1, 1)
        self.lab4_button = QtWidgets.QPushButton(self.centralwidget)
        self.lab4_button.setObjectName("lab4_button")
        self.gridLayout.addWidget(self.lab4_button, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.lab1_button.setEnabled(False)
        self.lab3_button.setEnabled(False)
        self.lab4_button.setEnabled(False)
        # self.lab1_button.clicked.connect(self.open_Lab1_Window)
        self.lab2_button.clicked.connect(self.open_Lab2_Window)
        #self.lab3_button.clicked.connect(self.open_Lab3_Window)
        #self.lab4_button.clicked.connect(self.open_Lab4_Window)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GTI411"))
        self.lab1_button.setText(_translate("MainWindow", "Lab1"))
        self.lab2_button.setText(_translate("MainWindow", "Lab 2"))
        self.lab3_button.setText(_translate("MainWindow", "Lab 3"))
        self.lab4_button.setText(_translate("MainWindow", "Lab 4"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
