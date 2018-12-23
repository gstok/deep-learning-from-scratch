#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path;
import pickle;
import gzip;
import numpy as np;
try:
    import urllib.request;
except ImportError:
    raise ImportError('You should use Python 3.x');
from PIL import Image;

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
    pilImg = Image.fromarray(np.uint8(img));
    pilImg.show();
    print(label);  

if (__name__ == "__main__"):
    result = getMnist(False, True, True);
    index = 99;

    img = result[1][0][index];
    label = result[1][1][index];
    showImgAndLabel(img, label);