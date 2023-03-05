# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Lab2_Window.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QDoubleValidator
from PyQt5.QtWidgets import QFileDialog
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class Ui_Lab2_Window(object):
    imageAdded = False
    src = np.zeros((200, 200, 4), np.uint8)
    isJpg = False
    isHighPass = False;

    def display_Image(self, fileName):
        self.src = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
        # afficher l'image originale
        pixmap = QPixmap(fileName)

        if(self.tabWidget.currentIndex()) == 0:
            self.label_3.setPixmap(pixmap)
        elif(self.tabWidget.currentIndex()) == 1:
            self.label_31.setPixmap(pixmap)
            self.label_32.clear()
            self.label_33.clear()
            self.label_25.clear()
            self.label_26.clear()
            self.label_27.clear()
        elif (self.tabWidget.currentIndex()) == 2:
            self.label_7.setPixmap(pixmap)
            self.label_8.clear()
            self.label_9.clear()
            self.label_13.clear()
            self.label_14.clear()
            self.label_15.clear()

        self.imageAdded = True


    def openImage(self):
        # read image from file dialog window
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.centralwidget, "Open Image", "", "Images (*.jpg);;Images (*.png);;All Files (*)", options=options)
        try:
            i = Image.open(fileName)
            if(i.format == 'JPEG'):
                self.isJpg = True
                self.display_Image(fileName)
            if(i.format == 'PNG'):
                self.isJpg = False
                self.display_Image(fileName)
            if(i.format != 'PNG' and (i.format != 'JPEG')):
                print('no valid type')
        except IOError:
            pass

    def filterChanged(self):
        if (self.comboBox_5.currentText() == 'Mean'):
            self.lineEdit.setText('0,11')
            self.lineEdit_2.setText('0,11')
            self.lineEdit_3.setText('0,11')
            self.lineEdit_4.setText('0,11')
            self.lineEdit_5.setText('0,11')
            self.lineEdit_6.setText('0,11')
            self.lineEdit_7.setText('0,11')
            self.lineEdit_8.setText('0,11')
            self.lineEdit_9.setText('0,11')
            self.lineEdit.setEnabled(False)
            self.lineEdit_2.setEnabled(False)
            self.lineEdit_3.setEnabled(False)
            self.lineEdit_4.setEnabled(False)
            self.lineEdit_5.setEnabled(False)
            self.lineEdit_6.setEnabled(False)
            self.lineEdit_7.setEnabled(False)
            self.lineEdit_8.setEnabled(False)
            self.lineEdit_9.setEnabled(False)
        else:
            self.lineEdit.setText('')
            self.lineEdit_2.setText('')
            self.lineEdit_3.setText('')
            self.lineEdit_4.setText('')
            self.lineEdit_5.setText('')
            self.lineEdit_6.setText('')
            self.lineEdit_7.setText('')
            self.lineEdit_8.setText('')
            self.lineEdit_9.setText('')
            self.lineEdit.setEnabled(True)
            self.lineEdit_2.setEnabled(True)
            self.lineEdit_3.setEnabled(True)
            self.lineEdit_4.setEnabled(True)
            self.lineEdit_5.setEnabled(True)
            self.lineEdit_6.setEnabled(True)
            self.lineEdit_7.setEnabled(True)
            self.lineEdit_8.setEnabled(True)
            self.lineEdit_9.setEnabled(True)

    def applyFilter(self):
        # Debut des modifs pour tester **********************************************************************************************************************
        if (self.imageAdded):
            filteredImage = any
            if (str(self.comboBox_5.currentText()) == 'Gaussian'):
                filteredImage = cv2.GaussianBlur(self.src, (5, 5),0)
            elif (str(self.comboBox_5.currentText()) == '4 - Neighbour Laplacian '):
                grayscale = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
                kernel = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
                filteredImage = cv2.filter2D(grayscale, -1, kernel)
            elif (str(self.comboBox_5.currentText()) == '8 - Neighbour Laplacian '):
                grayscale = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
                kernel = np.array([[0, 1, 0],
                                  [1, -8, 1],
                                  [0, 1, 0]])
                filteredImage = cv2.filter2D(grayscale, -1, kernel)
            elif (str(self.comboBox_5.currentText()) == 'Sobel Horiz'):
                grayscale = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
                sobelH = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, 3)
                seuil = 128
                filteredImage = np.zeros_like(sobelH)
                filteredImage[sobelH > seuil] = 255
            elif (str(self.comboBox_5.currentText()) == 'Sobel Vert'):
                grayscale = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
                sobelV = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, 3)
                seuil = 128
                filteredImage = np.zeros_like(sobelV)
                filteredImage[sobelV > seuil] = 255
            elif (str(self.comboBox_5.currentText()) == 'Sobel'):
                grayscale = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
                sobelH = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, 3)
                sobelV = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, 3)
                sobel = cv2.addWeighted(sobelH, 0.5, sobelV, 0.5, 0)
                seuil = 128
                filteredImage = np.zeros_like(sobel)
                filteredImage[sobel > seuil] = 255

            if (str(self.comboBox_7.currentText()) == 'Circular'):
                #jpg image has 3 channels
                if (self.isJpg == True):
                    filteredImage[0] = filteredImage[filteredImage.shape[0] - 1]  # ligne y = 0
                    filteredImage[filteredImage.shape[0] - 1] = filteredImage[0]  # ligne y = (height - 1)
                    filteredImage[:, 0] = filteredImage[:, filteredImage.shape[1] - 1]  # colonne x = 0
                    filteredImage[:, filteredImage.shape[1] - 1] = filteredImage[:, 0]  # colonne y = (width - 1)
                # png image has 4 channels
                if (self.isJpg == False):
                    filteredImage[0] = [0, 0, 0, 255]  # ligne y = 0
                    filteredImage[filteredImage.shape[0] - 1] = [0, 0, 0, 255]  # ligne y = (height - 1)
                    filteredImage[:, 0] = [0, 0, 0, 255]  # colonne x = 0
                    filteredImage[:, filteredImage.shape[1] - 1] = [0, 0, 0, 255]  # colonne y = (width - 1)

            if (str(self.comboBox_6.currentText()) == 'Normalize 0 to 255'):
                test = filteredImage
                filteredImage = cv2.normalize(test, None, 0, 255, cv2.NORM_MINMAX)

            # afficher l'image filtrée
            cv2.imwrite('blurred_image.jpg', filteredImage)
            pixmap = QPixmap('blurred_image.jpg')
            self.label_21.setPixmap(pixmap)
        # Fin des modifs pour tester **********************************************************************************************************************

        if (str(self.comboBox_5.currentText()) == 'Mean' and str(self.comboBox_7.currentText()) == '0' and str(
                self.comboBox_6.currentText()) == 'Clamp 0 ... 255' and self.imageAdded):
            # appliquer le filtre moyenneur 3*3
            blur = cv2.blur(self.src, (3, 3))
            # clamp 0 ... 255
            for i in range(len(blur)):
                for j in range(len(blur[0])):
                    pixel_b = blur[i][j][0]
                    pixel_g = blur[i][j][1]
                    pixel_r = blur[i][j][2]
                    if (pixel_b < 0 and pixel_g < 0 and pixel_r < 0):
                        blur[i][j] = [0, 0, 0]
                        if (self.isJpg == False): #image png
                            alpha = blur[i][j][3]
                            blur[i][j] = [0, 0, 0, alpha]
                        print('valeur inférieure à 0 trouvée')
                    if (pixel_b > 255 and pixel_g > 255 and pixel_r > 255):
                        blur[i][j] = [255, 255, 255]
                        if (self.isJpg == False): #image png
                            alpha = blur[i][j][3]
                            blur[i][j] = [255, 255, 255, alpha]
                        print('valeur supérieure à 255 trouvée')
            # border
            #jpg image has 3 channels
            if (self.isJpg == True):
                blur[0] = [0, 0, 0]  # ligne y = 0
                blur[blur.shape[0] - 1] = [0, 0, 0]  # ligne y = (height - 1)
                blur[:, 0] = [0, 0, 0]  # colonne x = 0
                blur[:, blur.shape[1] - 1] = [0, 0, 0]  # colonne y = (width - 1)
                cv2.imwrite('blurred_image.jpg', blur)
                # afficher l'image filtrée
                pixmap = QPixmap('blurred_image.jpg')
                self.label_21.setPixmap(pixmap)
            # png image has 4 channels
            if (self.isJpg == False):
                blur[0] = [0, 0, 0, 255]  # ligne y = 0
                blur[blur.shape[0] - 1] = [0, 0, 0, 255]  # ligne y = (height - 1)
                blur[:, 0] = [0, 0, 0, 255]  # colonne x = 0
                blur[:, blur.shape[1] - 1] = [0, 0, 0, 255]  # colonne y = (width - 1)
                cv2.imwrite('blurred_image.png', blur)
                # afficher l'image filtrée
                pixmap = QPixmap('blurred_image.png')
                self.label_21.setPixmap(pixmap)

    def applyCanny(self):
        if(self.imageAdded):
            blurred = cv2.GaussianBlur(self.src, (int(self.lineEdit_12.text()), int(self.lineEdit_12.text())), 0)
            nameFile = 'blurred.jpg' if self.isJpg else 'blurred.png'
            cv2.imwrite(nameFile, blurred)
            blurredPixmap = QPixmap(nameFile)
            self.label_25.setPixmap(blurredPixmap)

            grayAndBlurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

            sobelX = cv2.Sobel(grayAndBlurred, cv2.CV_64F, 1, 0, ksize=3)
            nameFile = 'sobelX.jpg' if self.isJpg else 'sobelX.png'
            cv2.imwrite(nameFile, sobelX)
            sobelXPixmap = QPixmap(nameFile)
            self.label_32.setPixmap(sobelXPixmap)

            sobelY = cv2.Sobel(grayAndBlurred, cv2.CV_64F, 0, 1, ksize=3)
            nameFile = 'sobelY.jpg' if self.isJpg else 'sobelY.png'
            cv2.imwrite(nameFile, sobelY)
            sobelYPixmap = QPixmap(nameFile)
            self.label_26.setPixmap(sobelYPixmap)


            # Find the gradient magnitude and direction
            mag = np.sqrt(sobelX ** 2 + sobelY ** 2)
            theta = np.arctan2(sobelY, sobelX)

            # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
            def non_max_suppression(img, D):
                # Get the image shape and create an output array
                M, N = img.shape
                out = np.zeros((M, N), dtype=np.int32)

                # Convert angle in degrees to radians
                angle = D * 180. / np.pi
                angle[angle < 0] += 180

                # Perform non-maximum suppression
                for i in range(1, M - 1):
                    for j in range(1, N - 1):
                        q = 255
                        r = 255

                        # Find the edge direction
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = img[i, j + 1]
                            r = img[i, j - 1]
                        elif 22.5 <= angle[i, j] < 67.5:
                            q = img[i + 1, j - 1]
                            r = img[i - 1, j + 1]
                        elif 67.5 <= angle[i, j] < 112.5:
                            q = img[i + 1, j]
                            r = img[i - 1, j]
                        elif 112.5 <= angle[i, j] < 157.5:
                            q = img[i - 1, j - 1]
                            r = img[i + 1, j + 1]

                        # Compare the intensity of the current pixel with its neighbors
                        if (img[i, j] >= q) and (img[i, j] >= r):
                            out[i, j] = img[i, j]
                        else:
                            out[i, j] = 0

                return out

            nms = non_max_suppression(mag, theta)

            nameFile = 'nms.jpg' if self.isJpg else 'nms.png'
            cv2.imwrite(nameFile, nms)
            nmsPixmap = QPixmap(nameFile)
            self.label_33.setPixmap(nmsPixmap)



            #https://theailearner.com/2019/05/22/canny-edge-detector/
            # Set high and low threshold
            highThreshold = int(self.lineEdit_14.text())
            lowThreshold = int(self.lineEdit_13.text())

            M, N = nms.shape
            out = np.zeros((M, N), dtype=np.uint8)

            # If edge intensity is greater than 'High' it is a sure-edge
            # below 'low' threshold, it is a sure non-edge
            strong_i, strong_j = np.where(nms >= highThreshold)
            zeros_i, zeros_j = np.where(nms < lowThreshold)

            # weak edges
            weak_i, weak_j = np.where((nms <= highThreshold) & (nms >= lowThreshold))

            # Set same intensity value for all edge pixels
            out[strong_i, strong_j] = 255
            out[zeros_i, zeros_j] = 0
            out[weak_i, weak_j] = 75

            # For weak edges,
            # if it is connected to a sure edge it will be considered as an edge otherwise suppressed.

            M, N = out.shape
            for i in range(1, M - 1):
                for j in range(1, N - 1):
                    if (out[i, j] == 75):
                        if 255 in [out[i + 1, j - 1], out[i + 1, j], out[i + 1, j + 1], out[i, j - 1], out[i, j + 1],
                                   out[i - 1, j - 1], out[i - 1, j], out[i - 1, j + 1]]:
                            out[i, j] = 255
                        else:
                            out[i, j] = 0

            nameFile = 'canny.jpg' if self.isJpg else 'canny.png'
            cv2.imwrite(nameFile, out)
            cannyPixmap = QPixmap(nameFile)
            self.label_27.setPixmap(cannyPixmap)

    def toggleLowHighPass(self):
        self.isHighPass = not self.isHighPass

        if self.isHighPass:
            self.label_20.setText("N parameter for High-Pass")
            self.label_5.setText("Ideal High-Pass reconstructed Image 1")
            self.label_11.setText("Ideal High-Pass Spectrum 1")
            self.label_6.setText("High-Pass Butterworth reconstructed Image 1")
            self.label_12.setText("High-Pass Butterworth Spectrum 1")
            self.pushButton.setText("Apply Ideal High-Pass Filter")
        else:
            self.label_20.setText("N parameter for Low-Pass")
            self.label_5.setText("Ideal Low-Pass reconstructed Image 1")
            self.label_11.setText("Ideal Low-Pass Spectrum 1")
            self.label_6.setText("Low-Pass Butterworth reconstructed Image 1")
            self.label_12.setText("Low-Pass Butterworth Spectrum 1")
            self.pushButton.setText("Apply Ideal Low-Pass Filter")

    #https://www.youtube.com/watch?v=C48AI4FvOKE&t=330s
    def applyIdealFilter(self):
        if self.imageAdded:
            gray_src = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

            F = np.fft.fft2(gray_src)
            Fshift = np.fft.fftshift(F)

            realF = np.log1p(np.abs(F))
            realFShift = np.log1p(np.abs(Fshift))

            nameFile = 'imagespectrum.jpg' if self.isJpg else 'imagespectrum.png'
            cv2.imwrite(nameFile, 20 * realFShift)
            pixmap = QPixmap(nameFile)
            self.label_13.setPixmap(pixmap)

            if self.isHighPass:
                # Filtre Passe-haut ideal
                M, N = gray_src.shape
                idealHP = np.zeros((M, N), dtype=np.float32)
                n = int(self.lineEdit_10.text())
                D0 = n * 7
                for u in range(M):
                    for v in range(N):
                        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
                        if D <= D0:
                            idealHP[u, v] = 0
                        else:
                            idealHP[u, v] = 1

                nameFile = 'idealhighpass.jpg' if self.isJpg else 'idealhighpass.png'
                cv2.imwrite(nameFile, 500 * idealHP)
                pixmap = QPixmap(nameFile)
                self.label_14.setPixmap(pixmap)

                Gshift = Fshift * idealHP
                G = np.fft.ifftshift(Gshift)
                g = np.abs(np.fft.ifft2(G))

                nameFile = 'idealhighpassfiltered.jpg' if self.isJpg else 'idealhighpassfiltered.png'
                cv2.imwrite(nameFile, g)
                pixmap = QPixmap(nameFile)
                self.label_8.setPixmap(pixmap)
            else:
                # Filtre Passe-bas ideal
                M, N = gray_src.shape
                idealLP = np.zeros((M, N), dtype=np.float32)
                n = int(self.lineEdit_10.text())
                D0 = n * 7
                for u in range(M):
                    for v in range(N):
                        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
                        if D <= D0:
                            idealLP[u, v] = 1
                        else:
                            idealLP[u, v] = 0

                nameFile = 'ideallowpass.jpg' if self.isJpg else 'ideallowpass.png'
                cv2.imwrite(nameFile, 500 * idealLP)
                pixmap = QPixmap(nameFile)
                self.label_14.setPixmap(pixmap)

                Gshift = Fshift * idealLP
                G = np.fft.ifftshift(Gshift)
                g = np.abs(np.fft.ifft2(G))

                nameFile = 'ideallowpassfiltered.jpg' if self.isJpg else 'ideallowpassfiltered.png'
                cv2.imwrite(nameFile, g)
                pixmap = QPixmap(nameFile)
                self.label_8.setPixmap(pixmap)

    #https://www.youtube.com/watch?v=C48AI4FvOKE&t=330s
    def applyButterworthFilter(self):
        if self.imageAdded:
            gray_src = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

            F = np.fft.fft2(gray_src)
            Fshift = np.fft.fftshift(F)

            realF = np.log1p(np.abs(F))
            realFShift = np.log1p(np.abs(Fshift))

            nameFile = 'imagespectrum.jpg' if self.isJpg else 'imagespectrum.png'
            cv2.imwrite(nameFile, 20 * realFShift)
            pixmap = QPixmap(nameFile)
            self.label_13.setPixmap(pixmap)

            if self.isHighPass:
                # Filtre Passe-haut Butterworth
                M, N = gray_src.shape
                butterworthHP = np.zeros((M, N), dtype=np.float32)
                n = int(self.lineEdit_11.text())
                D0 = n * 7
                for u in range(M):
                    for v in range(N):
                        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
                        butterworthHP[u, v] = 1 / (1 + (D0 / D) ** (2 * n))

                nameFile = 'butterworthhighpass.jpg' if self.isJpg else 'butterworthhighpass.png'
                cv2.imwrite(nameFile, 500 * butterworthHP)
                pixmap = QPixmap(nameFile)
                self.label_15.setPixmap(pixmap)

                Gshift = Fshift * butterworthHP
                G = np.fft.ifftshift(Gshift)
                g = np.abs(np.fft.ifft2(G))

                nameFile = 'butterworthhighpassfiltered.jpg' if self.isJpg else 'butterworthhighpassfiltered.png'
                cv2.imwrite(nameFile, g)
                pixmap = QPixmap(nameFile)
                self.label_9.setPixmap(pixmap)
            else:
                # Filtre Passe-bas Butterworth
                M, N = gray_src.shape
                butterworthLP = np.zeros((M, N), dtype=np.float32)
                n = int(self.lineEdit_11.text())
                D0 = n * 7
                for u in range(M):
                    for v in range(N):
                        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
                        butterworthLP[u, v] = 1 / (1 + (D / D0) ** (2 * n))

                nameFile = 'butterworthlowpass.jpg' if self.isJpg else 'butterworthlowpass.png'
                cv2.imwrite(nameFile, 500 * butterworthLP)
                pixmap = QPixmap(nameFile)
                self.label_15.setPixmap(pixmap)

                Gshift = Fshift * butterworthLP
                G = np.fft.ifftshift(Gshift)
                g = np.abs(np.fft.ifft2(G))

                nameFile = 'butterworthlowpassfiltered.jpg' if self.isJpg else 'butterworthlowpassfiltered.png'
                cv2.imwrite(nameFile, g)
                pixmap = QPixmap(nameFile)
                self.label_9.setPixmap(pixmap)


    def setupUi(self, Lab2_Window):
        Lab2_Window.setObjectName("Lab2_Window")
        #Lab2_Window.resize(832, 629)
        Lab2_Window.showMaximized()
        self.centralwidget = QtWidgets.QWidget(Lab2_Window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_3 = QtWidgets.QFrame(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 130))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_14.setObjectName("gridLayout_14")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_14.addItem(spacerItem, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_14.addWidget(self.pushButton_2, 0, 3, 1, 1)
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(300, 100))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_2.addWidget(self.lineEdit_2, 0, 1, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_6.setInputMask("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_2.addWidget(self.lineEdit_6, 1, 2, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_2.addWidget(self.lineEdit_7, 2, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 0, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_2.addWidget(self.lineEdit_5, 1, 1, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_2.addWidget(self.lineEdit_9, 2, 2, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_2.addWidget(self.lineEdit_3, 0, 2, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout_2.addWidget(self.lineEdit_8, 2, 1, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_2.addWidget(self.lineEdit_4, 1, 0, 1, 1)
        self.gridLayout_14.addWidget(self.frame_6, 0, 2, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.frame_2)
        self.formLayout_2.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_19 = QtWidgets.QLabel(self.frame_2)
        self.label_19.setObjectName("label_19")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.comboBox_7 = QtWidgets.QComboBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_7.sizePolicy().hasHeightForWidth())
        self.comboBox_7.setSizePolicy(sizePolicy)
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_7)
        self.label_18 = QtWidgets.QLabel(self.frame_2)
        self.label_18.setObjectName("label_18")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.comboBox_6 = QtWidgets.QComboBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_6.sizePolicy().hasHeightForWidth())
        self.comboBox_6.setSizePolicy(sizePolicy)
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox_6)
        self.comboBox_5 = QtWidgets.QComboBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_5.sizePolicy().hasHeightForWidth())
        self.comboBox_5.setSizePolicy(sizePolicy)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_5)
        self.label_17 = QtWidgets.QLabel(self.frame_2)
        self.label_17.setObjectName("label_17")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.gridLayout_14.addWidget(self.frame_2, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_14.addItem(spacerItem1, 0, 4, 1, 1)
        self.gridLayout_3.addWidget(self.frame_3, 0, 0, 1, 1)
        self.frame_5 = QtWidgets.QFrame(self.tab)
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame_4)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.frame_4)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.verticalLayout.addWidget(self.frame_4)
        self.frame = QtWidgets.QFrame(self.frame_5)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.label_21 = QtWidgets.QLabel(self.frame)
        self.label_21.setFrameShape(QtWidgets.QFrame.Box)
        self.label_21.setText("")
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_3.addWidget(self.label_21)
        self.verticalLayout.addWidget(self.frame)
        self.gridLayout_3.addWidget(self.frame_5, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.scrollbar_tab_2 = QtWidgets.QScrollArea(widgetResizable=True)
        self.scrollbar_tab_2.setWidget(self.tab_2)
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_20 = QtWidgets.QFrame(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_20.sizePolicy().hasHeightForWidth())
        self.frame_20.setSizePolicy(sizePolicy)
        self.frame_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_20)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 0, 0, 1, 1)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayout_4.setFormAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTop|QtCore.Qt.AlignTrailing)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_34 = QtWidgets.QLabel(self.frame_20)
        self.label_34.setObjectName("label_34")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_34)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_12.sizePolicy().hasHeightForWidth())
        self.lineEdit_12.setSizePolicy(sizePolicy)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_12)
        self.label_35 = QtWidgets.QLabel(self.frame_20)
        self.label_35.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_35.setObjectName("label_35")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_35)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.frame_20)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_13)
        self.label_36 = QtWidgets.QLabel(self.frame_20)
        self.label_36.setObjectName("label_36")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_36)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.frame_20)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_14)
        self.gridLayout_4.addLayout(self.formLayout_4, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem3, 0, 2, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_10.addWidget(self.pushButton_3)
        self.gridLayout_4.addLayout(self.verticalLayout_10, 0, 3, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem4, 0, 4, 1, 1)
        self.verticalLayout_9.addWidget(self.frame_20)
        self.frame_17 = QtWidgets.QFrame(self.tab_2)
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_17)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_18 = QtWidgets.QFrame(self.frame_17)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_18.sizePolicy().hasHeightForWidth())
        self.frame_18.setSizePolicy(sizePolicy)
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.frame_18)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_28 = QtWidgets.QLabel(self.frame_18)
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")
        self.gridLayout_12.addWidget(self.label_28, 0, 0, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.frame_18)
        self.label_29.setAlignment(QtCore.Qt.AlignCenter)
        self.label_29.setObjectName("label_29")
        self.gridLayout_12.addWidget(self.label_29, 0, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.frame_18)
        self.label_30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_30.setObjectName("label_30")
        self.gridLayout_12.addWidget(self.label_30, 0, 2, 1, 1)
        self.verticalLayout_8.addWidget(self.frame_18)
        self.frame_19 = QtWidgets.QFrame(self.frame_17)
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.frame_19)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.label_32 = QtWidgets.QLabel(self.frame_19)
        self.label_32.setFrameShape(QtWidgets.QFrame.Box)
        self.label_32.setText("")
        self.label_32.setObjectName("label_32")
        self.gridLayout_13.addWidget(self.label_32, 0, 1, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.frame_19)
        self.label_31.setFrameShape(QtWidgets.QFrame.Box)
        self.label_31.setText("")
        self.label_31.setObjectName("label_31")
        self.gridLayout_13.addWidget(self.label_31, 0, 0, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.frame_19)
        self.label_33.setFrameShape(QtWidgets.QFrame.Box)
        self.label_33.setText("")
        self.label_33.setObjectName("label_33")
        self.gridLayout_13.addWidget(self.label_33, 0, 2, 1, 1)
        self.verticalLayout_8.addWidget(self.frame_19)
        self.verticalLayout_9.addWidget(self.frame_17)
        self.frame_13 = QtWidgets.QFrame(self.tab_2)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_13)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.frame_15 = QtWidgets.QFrame(self.frame_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_15.sizePolicy().hasHeightForWidth())
        self.frame_15.setSizePolicy(sizePolicy)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.frame_15)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_16 = QtWidgets.QLabel(self.frame_15)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout_10.addWidget(self.label_16, 0, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.frame_15)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.gridLayout_10.addWidget(self.label_23, 0, 1, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.frame_15)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.gridLayout_10.addWidget(self.label_24, 0, 2, 1, 1)
        self.verticalLayout_7.addWidget(self.frame_15)
        self.frame_16 = QtWidgets.QFrame(self.frame_13)
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.frame_16)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_25 = QtWidgets.QLabel(self.frame_16)
        self.label_25.setFrameShape(QtWidgets.QFrame.Box)
        self.label_25.setText("")
        self.label_25.setObjectName("label_25")
        self.gridLayout_11.addWidget(self.label_25, 0, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.frame_16)
        self.label_26.setFrameShape(QtWidgets.QFrame.Box)
        self.label_26.setText("")
        self.label_26.setObjectName("label_26")
        self.gridLayout_11.addWidget(self.label_26, 0, 1, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.frame_16)
        self.label_27.setFrameShape(QtWidgets.QFrame.Box)
        self.label_27.setText("")
        self.label_27.setObjectName("label_27")
        self.gridLayout_11.addWidget(self.label_27, 0, 2, 1, 1)
        self.verticalLayout_7.addWidget(self.frame_16)
        self.verticalLayout_9.addWidget(self.frame_13)
        self.tabWidget.addTab(self.scrollbar_tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.scrollbar_tab_3 = QtWidgets.QScrollArea(widgetResizable=True)
        self.scrollbar_tab_3.setWidget(self.tab_3)
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_14 = QtWidgets.QFrame(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.frame_14)
        self.gridLayout_9.setObjectName("gridLayout_9")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem5, 0, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem6, 0, 3, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setFormAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.formLayout.setObjectName("formLayout")
        self.label_20 = QtWidgets.QLabel(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy)
        self.label_20.setMinimumSize(QtCore.QSize(145, 0))
        self.label_20.setObjectName("label_20")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_10.sizePolicy().hasHeightForWidth())
        self.lineEdit_10.setSizePolicy(sizePolicy)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_10)
        self.verticalLayout_5.addLayout(self.formLayout)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_22 = QtWidgets.QLabel(self.frame_14)
        self.label_22.setObjectName("label_22")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.frame_14)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_11)
        self.verticalLayout_5.addLayout(self.formLayout_3)
        self.gridLayout_9.addLayout(self.verticalLayout_5, 0, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem7, 0, 5, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton = QtWidgets.QPushButton(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_4.addWidget(self.pushButton)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_14)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_4.addWidget(self.pushButton_4)

        self.pushButton_5 = QtWidgets.QPushButton(self.frame_14)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_4.addWidget(self.pushButton_5)

        self.gridLayout_9.addLayout(self.verticalLayout_4, 0, 4, 1, 1)
        self.verticalLayout_6.addWidget(self.frame_14)
        self.frame_7 = QtWidgets.QFrame(self.tab_3)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_9.sizePolicy().hasHeightForWidth())
        self.frame_9.setSizePolicy(sizePolicy)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_9)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_4 = QtWidgets.QLabel(self.frame_9)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame_9)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame_9)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.frame_9)
        self.frame_10 = QtWidgets.QFrame(self.frame_7)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame_10)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_7 = QtWidgets.QLabel(self.frame_10)
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.gridLayout_8.addWidget(self.label_7, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame_10)
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.gridLayout_8.addWidget(self.label_8, 0, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.frame_10)
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.gridLayout_8.addWidget(self.label_9, 0, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.frame_10)
        self.verticalLayout_6.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.tab_3)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_12 = QtWidgets.QFrame(self.frame_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy)
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_12)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_10 = QtWidgets.QLabel(self.frame_12)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame_12)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_6.addWidget(self.label_11, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.frame_12)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 0, 2, 1, 1)
        self.verticalLayout_3.addWidget(self.frame_12)
        self.frame_11 = QtWidgets.QFrame(self.frame_8)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_11)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_13 = QtWidgets.QLabel(self.frame_11)
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.gridLayout_7.addWidget(self.label_13, 0, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame_11)
        self.label_14.setFrameShape(QtWidgets.QFrame.Box)
        self.label_14.setText("")
        self.label_14.setObjectName("label_14")
        self.gridLayout_7.addWidget(self.label_14, 0, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.frame_11)
        self.label_15.setFrameShape(QtWidgets.QFrame.Box)
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.gridLayout_7.addWidget(self.label_15, 0, 2, 1, 1)
        self.verticalLayout_3.addWidget(self.frame_11)
        self.verticalLayout_6.addWidget(self.frame_8)
        self.tabWidget.addTab(self.scrollbar_tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        Lab2_Window.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Lab2_Window)
        self.statusbar.setObjectName("statusbar")
        Lab2_Window.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(Lab2_Window)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 832, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuFilter = QtWidgets.QMenu(self.menuBar)
        self.menuFilter.setObjectName("menuFilter")
        Lab2_Window.setMenuBar(self.menuBar)
        self.actionExit = QtWidgets.QAction(Lab2_Window)
        self.actionExit.setObjectName("actionExit")
        self.actionAdd_Image = QtWidgets.QAction(Lab2_Window)
        self.actionAdd_Image.setObjectName("actionAdd_Image")
        self.actionApply_Filter = QtWidgets.QAction(Lab2_Window)
        self.actionApply_Filter.setObjectName("actionApply_Filter")
        self.menuFile.addAction(self.actionExit)
        self.menuFilter.addAction(self.actionAdd_Image)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuFilter.menuAction())

        self.label_3.setScaledContents(True)
        self.label_21.setScaledContents(True)
        self.lineEdit.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_2.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_3.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_4.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_5.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_6.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_7.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_8.setValidator(QDoubleValidator(999999, -999999, 8))
        self.lineEdit_9.setValidator(QDoubleValidator(999999, -999999, 8))

        # Connection des méthodes
        self.pushButton_2.clicked.connect(self.applyFilter)
        self.comboBox_5.currentIndexChanged.connect(self.filterChanged)
        self.actionAdd_Image.triggered.connect(self.openImage)
        self.pushButton_3.clicked.connect(self.applyCanny)
        self.pushButton_5.clicked.connect(self.toggleLowHighPass)
        self.pushButton.clicked.connect(self.applyIdealFilter)
        self.pushButton_4.clicked.connect(self.applyButterworthFilter)

        self.retranslateUi(Lab2_Window)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Lab2_Window)

    def retranslateUi(self, Lab2_Window):
        _translate = QtCore.QCoreApplication.translate
        Lab2_Window.setWindowTitle(_translate("Lab2_Window", "Lab2_Window"))
        self.pushButton_2.setText(_translate("Lab2_Window", "Apply Filter"))
        self.label_19.setText(_translate("Lab2_Window", "Handling Borders"))
        self.comboBox_7.setItemText(0, _translate("Lab2_Window", "0"))
        self.comboBox_7.setItemText(1, _translate("Lab2_Window", "None"))
        self.comboBox_7.setItemText(2, _translate("Lab2_Window", "Copy"))
        self.comboBox_7.setItemText(3, _translate("Lab2_Window", "Mirror"))
        self.comboBox_7.setItemText(4, _translate("Lab2_Window", "Circular"))
        self.label_18.setText(_translate("Lab2_Window", "Range"))
        self.comboBox_6.setItemText(0, _translate("Lab2_Window", "Clamp 0 ... 255"))
        self.comboBox_6.setItemText(1, _translate("Lab2_Window", "Abs and normalize to 255"))
        self.comboBox_6.setItemText(2, _translate("Lab2_Window", "Abs and normalize 0 to 255"))
        self.comboBox_6.setItemText(3, _translate("Lab2_Window", "Normalize 0 to 255"))
        self.comboBox_5.setItemText(0, _translate("Lab2_Window", "Costum"))
        self.comboBox_5.setItemText(1, _translate("Lab2_Window", "Mean"))
        self.comboBox_5.setItemText(2, _translate("Lab2_Window", "Gaussian"))
        self.comboBox_5.setItemText(3, _translate("Lab2_Window", "4 - Neighbour Laplacian "))
        self.comboBox_5.setItemText(4, _translate("Lab2_Window", "8 - Neighbour Laplacian "))
        self.comboBox_5.setItemText(5, _translate("Lab2_Window", "Sobel Horiz"))
        self.comboBox_5.setItemText(6, _translate("Lab2_Window", "Sobel Vert"))
        self.comboBox_5.setItemText(7, _translate("Lab2_Window", "Sobel"))
        self.label_17.setText(_translate("Lab2_Window", "Filter Type"))
        self.label.setText(_translate("Lab2_Window", "Original Image"))
        self.label_2.setText(_translate("Lab2_Window", "Filtered Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Lab2_Window", "Spatial Filters"))
        self.label_34.setText(_translate("Lab2_Window", "Gaussian Filter Size"))
        self.label_35.setText(_translate("Lab2_Window", "Min Threshold"))
        self.label_36.setText(_translate("Lab2_Window", "Max Threshold"))
        self.pushButton_3.setText(_translate("Lab2_Window", "Apply Filter"))
        self.label_28.setText(_translate("Lab2_Window", "Original Image"))
        self.label_29.setText(_translate("Lab2_Window", "Gradient X"))
        self.label_30.setText(_translate("Lab2_Window", "Local Maxima"))
        self.label_16.setText(_translate("Lab2_Window", "Smoothed Image"))
        self.label_23.setText(_translate("Lab2_Window", "Gradient Y"))
        self.label_24.setText(_translate("Lab2_Window", "Final Contour Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.scrollbar_tab_2), _translate("Lab2_Window", "Canny Algorithm"))
        self.label_20.setText(_translate("Lab2_Window", "N parameter for Low-Pass"))
        self.label_22.setText(_translate("Lab2_Window", "N parameter for Butterworth  "))

        self.pushButton.setText(_translate("Lab2_Window", "Apply Ideal Low-Pass Filter"))
        self.pushButton_4.setText(_translate("Lab2_Window", "Apply Butterworth Filter"))
        self.pushButton_5.setText(_translate("Lab2_Window", "Toggle Low/High Pass"))

        self.label_4.setText(_translate("Lab2_Window", "Original Image"))
        self.label_5.setText(_translate("Lab2_Window", " Ideal Low-Pass reconstructed Image 1"))
        self.label_6.setText(_translate("Lab2_Window", "Low-Pass Butterworth reconstructed Image 1"))
        self.label_10.setText(_translate("Lab2_Window", "Original Spectrum"))
        self.label_11.setText(_translate("Lab2_Window", "Ideal Low-Pass Spectrum 1"))
        self.label_12.setText(_translate("Lab2_Window", "Low-Pass Butterworth Spectrum 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.scrollbar_tab_3), _translate("Lab2_Window", "Frequency Filters"))
        self.menuFile.setTitle(_translate("Lab2_Window", "File"))
        self.menuFilter.setTitle(_translate("Lab2_Window", "Add"))
        self.actionExit.setText(_translate("Lab2_Window", "Exit"))
        self.actionAdd_Image.setText(_translate("Lab2_Window", "Add Image"))
        self.actionApply_Filter.setText(_translate("Lab2_Window", "Apply Filter"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Lab2_Window = QtWidgets.QMainWindow()
    ui = Ui_Lab2_Window()
    ui.setupUi(Lab2_Window)
    Lab2_Window.show()
    sys.exit(app.exec_())
