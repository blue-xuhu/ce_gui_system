import threading
import cv2
import DisplayUI
import sys
import DisplayUI0
from PyQt5.QtCore import QFile,Qt,QSize
from PyQt5.QtWidgets import QFileDialog, QMessageBox,QApplication, QMainWindow,QListWidget,QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import os
import torch
import torch.nn.functional as F
from tkinter import filedialog
from shizhan import Net
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
# import input1
from collections import defaultdict
import shutil
# import qdarkstyle
import argparse
import codecs
import logging
import os
import os.path as osp
import sys
import yaml
from basiclogger import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from labelme import __appname__
from labelme import __version__
from labelme.app import MainWindow
from labelme.config import get_config
from labelme.logger import logger
import pymysql as sql
from labelme.utils import newIcon
from efficientnet_pytorch import EfficientNet
from utils import Tester
class Display():
    def __init__(self, ui, mainWnd,ui1, mainWnd1,path,cl):
        _translate = QtCore.QCoreApplication.translate
        self.ui0 = ui
        self.mainWnd0 = mainWnd
        # 信号槽设置
        self.a=False
        self.ui0.Change.clicked.connect(self.Change)
        self.ui = ui1
        # print([self.ui.labelImg.x0,self.ui.labelImg.y0,self.ui.labelImg.x1,self.ui.labelImg.y1])
        self.mainWnd = mainWnd1
        self.imgPath=path
        self.classes=cl
        self.ui0.lineEdit_1.setPlaceholderText(_translate("MainWindow", "姓名"))
        self.ui0.lineEdit_2.setPlaceholderText(_translate("MainWindow", "姓氏"))
        self.ui0.lineEdit_3.setPlaceholderText(_translate("MainWindow", "名字"))
        self.ui0.lineEdit_4.setPlaceholderText(_translate("MainWindow", "年龄"))
        self.ui0.lineEdit_5.setPlaceholderText(_translate("MainWindow", "电话"))
        self.ui0.lineEdit_6.setPlaceholderText(_translate("MainWindow", "性别"))
        self.ui0.lineEdit_7.setPlaceholderText(_translate("MainWindow", "地址"))
        self.ui0.textEdit.setPlaceholderText(_translate("MainWindow", "病历"))
        self.ui0.pushButton.clicked.connect(self.CreateDB)

    def CreateDB(self):
        print(1)
        self.dbb = sql.Connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            passwd='root',
            db='banknh',
            # charset='utf8'
        )
        self.c = self.dbb.cursor()

        # c.execute(''' CREATE TABLE IF NOT EXISTS HISTORY(
        #         #        ID INT NOT NULL AUTO_INCREMENT,
        #         #        USERNAME VARCHAR(100) NOT NULL,
        #         #        FIRSTNAME VARCHAR(100) NOT NULL,
        #         #        LASTNAME VARCHAR(100) NOT NULL,
        #         #        AGE VARCHAR(100) NOT NULL,
        #         #        PHONE VARCHAR(100) NOT NULL,
        #         #        SEX VARCHAR(100),
        #         #        ADDRESS VARCHAR(100) NOT NULL);
        #         #        ''')
        print(2)
        self.insertdb()
        print(3)
        ######################################################
        ######   AUTHENTICATION FOR REGISTRATION PAGE  #######
        ######################################################
    def general_message(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Question)
        msg.exec_()
    def general_message1(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Question)
        msg.exec_()
    def insertdb(self):
        username = self.ui0.lineEdit_1.text()
        LOGGING('./日志')
        logging.info('患者名：'+username)

        firstname = self.ui0.lineEdit_2.text()
        lastname = self.ui0.lineEdit_3.text()
        age = self.ui0.lineEdit_4.text()
        # if '@' not in email:
        #     self.general_message('Invalid Email', 'Please Check your Email again')
        #     return email
        # else:
            # password = self.lineEdit_password.text()
            # confirmPass = self.lineEdit_confirmPassword.text()
            # if password != confirmPass:
            #     self.general_message('password Error', 'Password Not Match')
            #     return password and confirmPass
            # elif len(password) != len(confirmPass):
            #     self.general_message('password Error', 'password not Match')
            #     return password and confirmPass

        phone = self.ui0.lineEdit_5.text()
        if len(phone) != 11:
            self.general_message('Invalid Number', 'please Check your Phone number')
            return phone
        else:
            sex = self.ui0.lineEdit_6.text()
            address = self.ui0.lineEdit_7.text()
            print(2.2)
            history=self.ui0.textEdit.toPlainText()
            print(history)
            print(2.1)
            print(sex)
            self.c.execute("INSERT INTO HISTORY(USERNAME, FIRSTNAME, LASTNAME, AGE, PHONE, SEX ,ADDRESS,HISTORY)VALUES ('%s','%s','%s','%s','%s','%s','%s','%s')" % (
                        str(username), str(firstname), str(lastname), str(age),
                        str(phone), str(sex), str(address),str(history)))
            print('insert done')
            self.dbb.commit()
            self.dbb.close()
            self.general_message1('提示信息', '录入成功')
                # self.login()

    def main(self):
        self.mainWnd.close()
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--version', '-V', action='store_true', help='show version'
        )
        parser.add_argument(
            '--reset-config', action='store_true', help='reset qt config'
        )
        parser.add_argument(
            '--logger-level',
            default='info',
            choices=['debug', 'info', 'warning', 'fatal', 'error'],
            help='logger level',
        )
        parser.add_argument('filename', nargs='?', help='image or label filename')
        parser.add_argument(
            '--output',
            '-O',
            '-o',
            help='output file or directory (if it ends with .json it is '
                 'recognized as file, else as directory)'
        )
        default_config_file = os.path.join(os.path.expanduser('~'), '.labelmerc')
        parser.add_argument(
            '--config',
            dest='config',
            help='config file or yaml-format string (default: {})'.format(
                default_config_file
            ),
            default=default_config_file,
        )
        # config for the gui
        parser.add_argument(
            '--nodata',
            dest='store_data',
            action='store_false',
            help='stop storing image data to JSON file',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--autosave',
            dest='auto_save',
            action='store_true',
            help='auto save',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--nosortlabels',
            dest='sort_labels',
            action='store_false',
            help='stop sorting labels',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--flags',
            help='comma separated list of flags OR file containing flags',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--labelflags',
            dest='label_flags',
            help='yaml string of label specific flags OR file containing json '
                 'string of label specific flags (ex. {person-\d+: [male, tall], '
                 'dog-\d+: [black, brown, white], .*: [occluded]})',  # NOQA
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--labels',
            help='comma separated list of labels OR file containing labels',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--validatelabel',
            dest='validate_label',
            choices=['exact'],
            help='label validation types',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--keep-prev',
            action='store_true',
            help='keep annotation of previous frame',
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--epsilon',
            type=float,
            help='epsilon to find nearest vertex on canvas',
            default=argparse.SUPPRESS,
        )
        args = parser.parse_args()

        if args.version:
            print('{0} {1}'.format(__appname__, __version__))
            sys.exit(0)

        logger.setLevel(getattr(logging, args.logger_level.upper()))

        if hasattr(args, 'flags'):
            if os.path.isfile(args.flags):
                with codecs.open(args.flags, 'r', encoding='utf-8') as f:
                    args.flags = [l.strip() for l in f if l.strip()]
            else:
                args.flags = [l for l in args.flags.split(',') if l]

        if hasattr(args, 'labels'):
            if os.path.isfile(args.labels):
                with codecs.open(args.labels, 'r', encoding='utf-8') as f:
                    args.labels = [l.strip() for l in f if l.strip()]
            else:
                args.labels = [l for l in args.labels.split(',') if l]

        if hasattr(args, 'label_flags'):
            if os.path.isfile(args.label_flags):
                with codecs.open(args.label_flags, 'r', encoding='utf-8') as f:
                    args.label_flags = yaml.safe_load(f)
            else:
                args.label_flags = yaml.safe_load(args.label_flags)

        config_from_args = args.__dict__
        config_from_args.pop('version')
        reset_config = config_from_args.pop('reset_config')
        filename = config_from_args.pop('filename')
        output = config_from_args.pop('output')
        config_file_or_yaml = config_from_args.pop('config')
        config = get_config(config_file_or_yaml, config_from_args)

        if not config['labels'] and config['validate_label']:
            logger.error('--labels must be specified with --validatelabel or '
                         'validate_label: true in the config file '
                         '(ex. ~/.labelmerc).')
            sys.exit(1)

        output_file = None
        output_dir = None
        if output is not None:
            if output.endswith('.json'):
                output_file = output
            else:
                output_dir = output

        translator = QtCore.QTranslator()
        translator.load(
            QtCore.QLocale.system().name(),
            osp.dirname(osp.abspath(__file__)) + '/translate'
        )
        # app = QtWidgets.QApplication(sys.argv)
        # app.setApplicationName(__appname__)
        # app.setWindowIcon(newIcon('icon'))
        # app.installTranslator(translator)
        win = MainWindow(
            config=config,
            filename=filename,
            output_file=output_file,
            output_dir=output_dir,
        )

        if reset_config:
            logger.info('Resetting Qt config: %s' % win.settings.fileName())
            win.settings.clear()
            sys.exit(0)

        win.show()
        win.raise_()
        # sys.exit(app.exec_())
    def Change(self):
        self.fileName, self.fileType = QFileDialog.getOpenFileName(self.mainWnd0, 'Choose file', '', '*.mp4')
        self.cap = cv2.VideoCapture(self.fileName)
        i=0
        while True:
            (flag,fram)=self.cap.read()
            if  fram is None:
                print(fram)
                break
            else:
                i+=1
                fileName='./1/image'+str(i)+'.jpg'
                cv2.imwrite(fileName,fram)
        self.mainWnd0.close()
        # self.ui.allFiles.resize(180, 700)
        self.allImgs = os.listdir(self.imgPath)  # 遍历路径，将所有文件放到列表框中
        print(self.allImgs)
        # m1=0
        print(2)
        self.a=self.shibie1()
        for imgTmp in self.allImgs:
            self.ui.allFiles.addItem(imgTmp)
            # l = self.shibie(self.imgPath + imgTmp)
            # print(l)
            # print(a[imgTmp])
            self.ui.allFiles_2.addItem(str(self.a[imgTmp]))
            # m1=max(m1,len(imgTmp))
        # self.ui.allFiles.resize(m1,self.ui.allFiles.height())
        # self.ui.allFiles.setMaximumSize(QSize(m1,16777215))
        # self.ui.allFiles.setMinimumSize(QSize(0,0))
        for index in range(self.ui.allFiles_2.count()):
            item = self.ui.allFiles_2.item(index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Unchecked)
            # print(item.checkState())
        print(2)
        print(self.ui.allFiles.count())
        self.ui.allFiles.itemClicked.connect(self.itemClick)
        self.ui.pushButton.clicked.connect(self.classify)
        print(1)
        self.ui.pushButton_2.clicked.connect(self.main)
        # self.ui.allFiles_2.itemDoubleClicked.connect(self.itemClickDouble)
        print(3)
        self.mainWnd.show()

        # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上

    def copyfile(self,srcfile, dstfile):  # 用来复制文件，源文件会保留
        if not os.path.isfile(srcfile):
            print("%s not exist!" % srcfile)
        else:
            f_path, f_name = os.path.split(dstfile)  # 分离文件名和路径
            if not os.path.exists(f_path):
                os.makedirs(f_path)  # 创建路径
            shutil.copyfile(srcfile, dstfile)  # 复制文件
            print("copy %s -> %s" % (srcfile, dstfile))

    def dt(self,s):
        from detect import detect
        import argparse
        import torch
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='config/yolov3.cfg', help='*.cfg path')
        parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
        parser.add_argument('--weights', type=str, default='weights/last.pt', help='weights path')
        parser.add_argument('--source', type=str, default=s,
                            help='source')  # input file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
        parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        opt = parser.parse_args()
        print(opt)
        with torch.no_grad():
            detect(opt)
    def classify(self):

        r = defaultdict(list)
        for index in range(self.ui.allFiles_2.count()):
            item = self.ui.allFiles_2.item(index)
            if item.checkState()!=0:
                # print(item.text())
                r[item.text()].append(index)
                # print(r[item.text()])
            # print(item.checkState())#打印状态
            # print(item.text())#打印文本
        # print(r['cat'])
        for i in r:
            print(i)
            print(r[i])
            for j in range(r[i][0],r[i][1]+1):
                current_img_path = self.allImgs[j]
                print(current_img_path)
                dir_path = i + "/"
                self.copyfile(self.imgPath + current_img_path, dir_path + current_img_path)

        self.dt(dir_path)
    def itemClick(self):  #列表框单击事件
        tmp = self.imgPath + self.ui.allFiles.currentItem().text()  #图像的绝对路径
        print(tmp)
        # self.ui.label.setText(self.shibie(tmp))
        self.ui.label.setText(str(self.a[self.ui.allFiles.currentItem().text()]))
        imgOri = cv2.imread(str(tmp),1)      #读取图像
        height = imgOri.shape[0]             #图像高度
        width = imgOri.shape[1]           # 计算图像宽度，缩放图像
        ratioY = self.ui.labelImg.height()/(height+0.0)  #按高度值缩放
        ratioX = self.ui.labelImg.width() / (width + 0.0)  # 按高度值缩放
        r=min(ratioY,ratioX)
        # height2 = self.ui.labelImg.height()
        # width2 = int(width*ratioY + 0.5)
        height2=int(height*r)
        width2=int(width*r)
        img2 = cv2.resize(imgOri, (width, height))
        cv2.imwrite("F:/tmp.jpg", img2)    # 将图像保存到本地
        qImgT = QPixmap("F:/tmp.jpg")   #读取本地的图像
        self.ui.labelImg.setPixmap(qImgT)         #标签显示图像
    def itemClickDouble(self):
        self.ui.allFiles_2.currentItem().setText('dsds')
        # input2.show()
        # self.ui.allFiles.setCurrentItem('dd')
        # QListWidget.openPersistentEditor(self,self.ui.allFiles.currentItem())
    def shibie(self,k):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('model.pth')  # 加载模型
        model = model.to(device)
        model.eval()  # 把模型转为test模式

        img = cv2.imread(k)  # 读取要预测的图片
        img = cv2.resize(img, (32, 32))
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        # 扩展后，为[1，1，28，28]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是10个分类的概率
        value, predicted = torch.max(output.data, 1)
        pred_class = self.classes[predicted.item()]
        return pred_class
    def shibie1(self):
        params = Tester.TestParams()
        params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
        params.ckpt = './ckpt_epoch_100.pth'  # './models/ckpt_epoch_400_res34.pth'
        flag = 1
        if flag == 1:
            params.testdata_dir = './1'
        if flag == 2:
            params.testdata_dir = './images/test/'

        # models
        # model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
        # model.fc = nn.Linear(512, 6)
        # model = resnet101(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
        # model.fc = nn.Linear(512*4, 6)
        model = EfficientNet.from_name('efficientnet-b0')
        model._fc = torch.nn.Linear(320 * 4, 4)

        # Test
        tester = Tester(model, params)
        a = {}
        tester.test(a)
        print('over')
        return a




if __name__ == '__main__':
    # input2=input1.Ui_Form()
    imgPath = "./1/"
    classes = ('plane', 'car',
               'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    app = QApplication(sys.argv)
    mainWnd0 = QMainWindow()
    mainWnd = QMainWindow()
    ui0 = DisplayUI0.Ui_MainWindow()
    ui = DisplayUI.Ui_MainWindow()
    ui.setupUi(mainWnd)
    # 可以理解成将创建的 ui 绑定到新建的 mainWnd 上
    ui0.setupUi(mainWnd0)
    display = Display(ui0, mainWnd0,ui,mainWnd,imgPath,classes)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainWnd0.show()
    # display1 = Display1(ui, mainWnd)
    sys.exit(app.exec_())
