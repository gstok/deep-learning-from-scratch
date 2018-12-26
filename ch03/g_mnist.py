#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path;
import pickle;
import gzip;
import time;
import numpy as np;
try:
    import urllib.request;
except ImportError:
    raise ImportError('You should use Python 3.x');
from PIL import Image;

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
    if (x.ndim == 1):
        t = t.reshape(1, t.size);
        x = x.reshape(1, x.size);
    batchSize = x.shape[0];
    return -np.sum(t * np.log(x + 1e-7)) / batchSize;


# 数值微分求导，求导仅仅对单个自变量求导
def numericalDiff (f, x):
    h = 1e-10;
    return (f(x + h) - f(x - h)) / (2 * h);

def numericalGradient (f, x):
    print(1);


def func (x, y):
    return x ** 2 + y ** 2;

def func1 (x):
    return func(x, 4);
def func2 (y):
    return func(3, y);


if (__name__ == "__main__"):
    result = getMnist(True, True, True);
    x = 3;
    y = 4;
    print(numericalDiff(func1, x));
    print(numericalDiff(func2, y));
