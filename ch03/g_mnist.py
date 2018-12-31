#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path;
import pickle;
import gzip;
import time;
import numpy as np;
import sys, os;
try:
    import urllib.request;
except ImportError:
    raise ImportError("You should use Python 3.x");
from PIL import Image;
sys.path.append(os.pardir);
from common.gradient import numerical_gradient as numericalGradient;

#让numpy不以科学计数法输出结果，否则看着太费劲了
np.set_printoptions(suppress = True);

mnistBaseUrl = "http://yann.lecun.com/exdb/mnist/";
fileMap = {
    "trainImg": "train-images-idx3-ubyte.gz",
    "trainLabel": "train-labels-idx1-ubyte.gz",
    "testImg": "t10k-images-idx3-ubyte.gz",
    "testLabel": "t10k-labels-idx1-ubyte.gz",
};
downDir = "../mnist/";
pklFile = "mnist.pkl";

# 下载mnist数据集到指定目录
def downloadMnist (baseUrl, fileMap, downDir):
    for key in fileMap.keys():
        value = fileMap[key];
        downUrl = "%s%s" % (baseUrl, value);
        downPath = "%s%s.gz" % (downDir, key);   
        if os.path.exists(downPath):
            pass;
            # print("%s 已经存在无需下载" % downUrl);
        else:
            # print("下载 %s 到 %s..." % (downUrl, downPath));
            urllib.request.urlretrieve(downUrl, downPath);
    return downDir;

# 加载图像数据方法
def loadImg (imgPath):
    with gzip.open(imgPath) as f:
        fileBuf = f.read();
        data = np.frombuffer(fileBuf, np.uint8, offset = 16);
    data = data.reshape(-1, 784);
    # print(data.shape);
    return data;

# 加载标签数据方法
def loadLabel (labelPath):
    with gzip.open(labelPath) as f:
        fileBuf = f.read();
        data = np.frombuffer(fileBuf, np.uint8, offset = 8);
    # print(data.shape);
    return data;

# 加载mnist数据集
def loadMnist (downDir, fileMap, pklFile):
    result = { };
    pklPath = "%s%s" % (downDir, pklFile);
    if (os.path.exists(pklPath)):
        with open(pklPath, "rb") as f:
            result = pickle.load(f);
    else:
        for key in fileMap:
            filePath = "%s%s.gz" % (downDir, key);
            if (filePath.endswith("Img.gz")):
                result[key] = loadImg(filePath);
            elif (filePath.endswith("Label.gz")):
                result[key] = loadLabel(filePath);
        with open(pklPath, "wb") as f:
            pickle.dump(result, f, -1);
    return result;

# 修改标签为OneHot模式
def changeOneHotLabel (data):
    newData = np.zeros((data.size, 10))
    for idx, row in enumerate(newData):
        row[data[idx]] = 1;
    return newData;

# 对外开放的获取Mnist接口
def getMnist (normalize = True, flatten = True, oneHotLabel = False):
    downloadMnist(mnistBaseUrl, fileMap, downDir);
    result = loadMnist(downDir, fileMap, pklFile);
    if (normalize):
        result["trainImg"] = result["trainImg"].astype(np.float);
        result["trainImg"] /= 255.0;
        result["testImg"] = result["testImg"].astype(np.float);
        result["testImg"] /= 255.0;
    if (not flatten):
        result["trainImg"] = result["trainImg"].reshape(-1, 1, 28, 28);
        result["testImg"] = result["testImg"].reshape(-1, 1, 28, 28);
    if (oneHotLabel):
        result["trainLabel"] = changeOneHotLabel(result["trainLabel"]);
        result["testLabel"] = changeOneHotLabel(result["testLabel"]);
    return (
        (
            result["trainImg"],
            result["trainLabel"]
        ),
        (
            result["testImg"],
            result["testLabel"]
        ),
    );

# 显示图片和标签方法
def showImgAndLabel (img, label):
    img = img.reshape((28, 28));
    print(label);
    pilImg = Image.fromarray(np.uint8(img));
    pilImg.show();

# sigmoid激活函数
def sigmoid (x):
    return 1 / (1 + np.exp(-x));

# softmax函数，此函数的作用是取0~1的概率分布
def softmax (x):
    c = np.max(x);
    expA = np.exp(x - c);
    sumExpA = np.sum(expA);
    y = expA / sumExpA;
    return y;

# 读取网络数据
def initNetwork ():
    network = { };
    with open("../mnist/sample_weight.pkl", "rb") as f:
        network = pickle.load(f);
    return network;

# 利用模型对图像进行分类预测
def predict (img, network):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"];
    b1, b2, b3 = network["b1"], network["b2"], network["b3"];
    a1 = np.dot(img, w1) + b1;
    z1 = sigmoid(a1);
    a2 = np.dot(z1, w2) + b2;
    z2 = sigmoid(a2);
    a3 = np.dot(z2, w3) + b3;
    y = softmax(a3);
    return y;

# 用测试数据验证模型精度，返回测试精度
def accuracy (imgs, labels, network):
    success  = 0;
    for index, img in enumerate(imgs):
        y = predict(img, network);
        p = np.argmax(y);
        if (p == labels[index]):
            success += 1;
    return float(success) / len(imgs);

# 使用批处理的方法对于测试数据验证精度
def accuracyBatch (imgs, labels, network):
    batchSize = 1000;
    success  = 0;
    for index in range(0, len(imgs), batchSize):
        imgsBat = imgs[index : index + batchSize];
        labelsBat = labels[index : index + batchSize];
        y = predict(imgsBat, network);
        p = np.argmax(y, axis = 1);
        b = p == labelsBat;
        success += np.sum(b);
    return float(success) / len(imgs);

# 损失函数之交叉熵函数
# 输入为0 ~ 1，这个值是softmax函数返回的概率值
# 输出为-无穷大 ~ 0，这个值是交叉熵，用来做损失函数的值
# 输入越接近0，输出越大，输入越接近1，输出越接近0
def crossEntropyError (x, t):
    delta = 1e-7;
    return -np.sum((t * np.log(x + delta)));

# miniBatch计算交叉熵
def crossEntropyErrorMB (x, t):
    # 变换参数格式，相当于套了一个维度，变成了二维数组
    if (x.ndim == 1):
        t = t.reshape(1, t.size);
        x = x.reshape(1, x.size);
    # 获取参数第一维大小
    batchSize = x.shape[0];
    # 这里相当于计算交叉熵的平均数
    return -np.sum(t * np.log(x + 1e-7)) / batchSize;


# 数值微分求导，求导仅仅对单个自变量求导
def numericalDiff (f, x):
    h = 1e-10;
    return (f(x + h) - f(x - h)) / (2 * h);

def func (npay):
    return np.sum(npay ** 2);

def numericalGradientLine (f, x):
    h = 1e-10;
    grad = np.zeros_like(x);
    for i in range(x.size):
        bak = x[i];
        x[i] = bak + h;
        y2 = f(x);
        x[i] = bak - h;
        y1 = f(x);
        x[i] = bak;
        grad[i] = (y2 - y1) / (2 * h);
    return grad;

# 可对于矩阵求梯度的函数
# def numericalGradient (f, x):
#     if (x.ndim == 1):
#         return numericalGradientLine(f, x);
#     else:
#         grad = np.zeros_like(x);
#         for index, line in enumerate(x):
#             grad[x] = numericalGradientLine(f, line);
#     return grad;


def gradientDescent (f, initX, lr = 0.0001, step = 100):
    x = initX;
    for i in range(step):
        grad = numericalGradient(f, x);
        x -= x * lr;
    return x;


class simpleNet:
    def __init__ (self):
        # self.w = np.array([
        #     [-0.30129775, 0.63226888, 0.29906706],
        #     [0.10313986, -1.6204596, -1.56058255],
        # ]);
        self.w = np.array([
            [-0.63929169, 0.59491399, 0.67441589],
            [-1.4568322, -1.79286679, 0.1717967],
        ]);
    def predict (self, x):
        return np.dot(x, self.w);
    def loss (self, x, t):
        y = self.predict(x);
        y = softmax(y);
        loss = crossEntropyErrorMB(y, t);
        print(y);
        return loss;
        




if (__name__ == "__main__"):
    result = getMnist(True, True, True);
    net = simpleNet();
    x = np.array([1.3, 6.0]);
    t = np.array([0, 0, 1]);
    net.loss(x, t);
    def funcNet (w):
        return net.loss(x, t);
    for i in range(100):
        grad = numericalGradient(funcNet, net.w);
        net.w -= grad * 0.01;
        # print(grad);
    print(net.w);
