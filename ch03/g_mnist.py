#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path;
try:
    import urllib.request;
except ImportError:
    raise ImportError('You should use Python 3.x');

mnistBaseUrl = "http://yann.lecun.com/exdb/mnist/";
fileMap = {
    "trainImg": "train-images-idx3-ubyte.gz",
    "trainLabel": "train-labels-idx1-ubyte.gz",
    "testImg": "t10k-images-idx3-ubyte.gz",
    "testLabel": "t10k-labels-idx1-ubyte.gz",
};

# 下载mnist数据集到指定目录
def downloadMnist (baseUrl, fileMap, downDir):
    for key in fileMap.keys():
        value = fileMap[key];
        downUrl = "%s%s" % (baseUrl, value);
        downPath = "%s%s.gz" % (downDir, key);   
        if os.path.exists(downPath):
            print("%s 已经存在无需下载" % downUrl);
        else:
            print("下载 %s 到 %s..." % (downUrl, downPath));
            urllib.request.urlretrieve(downUrl, downPath);
    return downDir;


downloadMnist(mnistBaseUrl, fileMap, "../mnist/");
