import numpy as np
import os
from scipy import ndimage, misc

DATA_DIR_PATH = '../data'

files = {}
for fileName in os.listdir(DATA_DIR_PATH):
    print fileName
    fileId, fileExtension = fileName.split('.')
    fileId = fileId[:-2] # removes _c, _d
    if fileId not in files:
        files[fileId] = {}
    filePath = os.path.join(DATA_DIR_PATH, fileName)
    try:
        files[fileId][fileExtension] = misc.imread(filePath)
    except:
        strData = np.array(open(filePath).read().split())
        files[fileId][fileExtension] = strData.astype(np.float)

print files
