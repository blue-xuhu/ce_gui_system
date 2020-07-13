# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DisplayUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5 import QtCore, QtGui, QtWidgets
class MyLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    #鼠标点击事件
    def mousePressEvent(self,event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
    #鼠标释放事件
    def mouseReleaseEvent(self,event):
        self.flag = False
    #鼠标移动事件
    def mouseMoveEvent(self,event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
    #绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        rect =QRect(self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
        painter.drawRect(rect)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(934, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.allFiles = QtWidgets.QListWidget(self.centralwidget)
        self.allFiles.setEnabled(True)
        self.allFiles.setMinimumSize(QtCore.QSize(120, 0))
        self.allFiles.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.allFiles.setObjectName("allFiles")
        self.horizontalLayout.addWidget(self.allFiles)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(60, 0))
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.allFiles_2 = QtWidgets.QListWidget(self.centralwidget)
        self.allFiles_2.setEnabled(True)
        self.allFiles_2.setMinimumSize(QtCore.QSize(80, 0))
        self.allFiles_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.allFiles_2.setObjectName("allFiles_2")
        self.horizontalLayout.addWidget(self.allFiles_2)
        self.labelImg = MyLabel(self.centralwidget)
        self.labelImg.setText("")
        self.labelImg.setObjectName("labelImg")
        self.labelImg.setMinimumSize(QtCore.QSize(0, 0))
        self.labelImg.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.horizontalLayout.addWidget(self.labelImg)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 100)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 934, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "save"))
        self.pushButton_2.setText(_translate("MainWindow", "detection"))
        self.menu.setTitle(_translate("MainWindow", "Type Here"))
