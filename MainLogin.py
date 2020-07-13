# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
# from MainProfile import Ui_MainWindow
from PyQt5.QtWidgets import QMessageBox
import os
import qdarkstyle
from vse import Display
import DisplayUI2
import DisplayUI0
from basiclogger import *
from shizhan import Net
class Ui_LoginWindow(object):
    def __init__(self,mainWnd0):
        self.mainWnd0=mainWnd0
    def beginLogin(self, LoginWindow):
        self.LoginWindow=LoginWindow
        self.LoginWindow.setObjectName("LoginWindow")
        self.LoginWindow.resize(453, 466)
        self.LoginWindow.setStyleSheet("background-color: rgb(12, 31, 45);\n"
"")
        self.centralwidget = QtWidgets.QWidget(self.LoginWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";\n"
"color: rgba(240, 240, 240, 240);")
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";\n"
"color: rgba(240, 240, 240, 240);")
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.pushButton_Login = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Login.setStyleSheet("color: rgb(240, 240, 240);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(78, 78, 78);")
        self.pushButton_Login.setObjectName("pushButton_Login")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.pushButton_Login)
        self.pushButton_Sign_up = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Sign_up.setStyleSheet("color: rgb(240, 240, 240);\n"
"font: 75 10pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(78, 78, 78);")
        self.pushButton_Sign_up.setObjectName("pushButton_Sign_up")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.pushButton_Sign_up)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("../../../Downloads/login_avater_9RD_icon.ico"))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_4)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.LoginWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self.LoginWindow)
        self.statusbar.setObjectName("statusbar")
        self.LoginWindow.setStatusBar(self.statusbar)

        self.retranslateUi(self.LoginWindow)
        QtCore.QMetaObject.connectSlotsByName(self.LoginWindow)

        ####################################################
        #         Connecting welcome page button 
        ####################################################

        self.pushButton_Login.clicked.connect(self.loginLogin)
        self.pushButton_Sign_up.clicked.connect(self.reg)

    def general_message(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Question)
        msg.exec_()


    def profile(self):
        # self.MainWindow = QtWidgets.QMainWindow()
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self.MainWindow)
        # self.MainWindow.show()
        self.LoginWindow.close()
        # os.system("python vse.py")
        # app = QtWidgets.QApplication(sys.argv
        # mainWnd0 = QtWidgets.QMainWindow()
        # mainWnd = QtWidgets.QMainWindow()
        # ui0 = DisplayUI0.Ui_MainWindow()
        # ui = DisplayUI.Ui_MainWindow()
        # ui.setupUi(mainWnd)
        # # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
        # ui0.setupUi(mainWnd0)
        # display = Display(ui0, mainWnd0, ui, mainWnd)
        print(121212)
        self.mainWnd0.show()
        # print(1)
        # sys.exit(app.exec_())
    def loginLogin(self):
        import pymysql as sql
        dbb = sql.Connect(
                            host='127.0.0.1',
                            port=3306,
                            user='root',
                            passwd='root',
                            db='banknh',
                            # charset='utf8'
                        )
        cur = dbb.cursor()
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        LOGGING('./日志')
        logging.info('用户名：'+username)

        print(username,password)
        cur.execute("select * from NEWBANK WHERE USERNAME = '%s' AND PASSWORD = '%s'" % (str(username),str(password)))
        result = cur.fetchall()

        if result:
            self.profile()
        else:
            self.general_message('User Error', 'User does not Exist')


    def reg(self):
        from registrationNew import Ui_registrationPage
        self.general_message('Back', 'Will you like to go Back')
        self.registrationPage = QtWidgets.QMainWindow()
        self.ui = Ui_registrationPage()
        self.ui.setupUi(self.registrationPage)
        self.registrationPage.show()
        
        

    def retranslateUi(self, LoginWindow):
        _translate = QtCore.QCoreApplication.translate
        LoginWindow.setWindowTitle(_translate("LoginWindow", "Login Page"))
        self.label_2.setText(_translate("LoginWindow", "UserName"))
        self.label_3.setText(_translate("LoginWindow", "PassWord"))
        self.pushButton_Login.setText(_translate("LoginWindow", "LOGIN"))
        self.pushButton_Sign_up.setText(_translate("LoginWindow", "BACK TO SIGN-UP PAGE"))


if __name__ == "__main__":
    imgPath = "1/"
    classes = ('plane', 'car',
               'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWnd0 = QtWidgets.QMainWindow()
    mainWnd = QtWidgets.QMainWindow()
    ui0 = DisplayUI0.Ui_MainWindow()
    ui1 = DisplayUI2.Ui_MainWindow()
    ui1.setupUi(mainWnd)
    ui0.setupUi(mainWnd0)
    display = Display(ui0, mainWnd0, ui1, mainWnd, imgPath, classes)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    LoginWindow = QtWidgets.QMainWindow()
    ui = Ui_LoginWindow(mainWnd0)
    ui.beginLogin(LoginWindow)
    LoginWindow.show()
    sys.exit(app.exec_())

