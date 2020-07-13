from torch import nn
from utils import Tester
from network import resnet34, resnet101
from efficientnet_pytorch import EfficientNet
# Set Test parameters
params = Tester.TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_100.pth'  #'./models/ckpt_epoch_400_res34.pth'
flag=1
if flag==1:
    params.testdata_dir = './images/train/'
if flag==2:
    params.testdata_dir = './images/test/'

# models
# model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
# model = resnet101(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
# model.fc = nn.Linear(512*4, 6)
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(320*4, 4)

# Test
tester = Tester(model, params)
a={}
tester.test(a)

if flag==1:
    f=open('./images/train.txt','r')
if flag==2:
    f = open('./images/test.txt', 'r')
lines = f.readlines()

# print(len(lines))
count = 0
from collections import defaultdict

d = defaultdict(int)
for i in range(len(lines)):
    #     print(len(lines[i]))
    if flag==1:
        s = lines[i][15:25]
        r = lines[i][26:28]
    if flag==2:
        s = lines[i][14:24]
        r = lines[i][25:27]
    d[r] += 1
    if a[s] == int(r):
        count += 1
    else:
        print(s, a[s], r)

print('ac=', count / len(lines))
# print(d)
#     print(r)

# %%

# print(count)

