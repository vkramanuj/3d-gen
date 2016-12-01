import numpy as np
import os
from scipy import ndimage, misc
import sys

VERBOSE = False

DATA_DIR_PATH = '../data'

files = {}
phogs = {}

class Image(object):
    def __init__(self, fileName, rgb_image, phog):
        self.fileName = fileName
        self.rgb_image = rgb_image
        self.phog = phog

    def __str__(self):
        string = "IMAGE[name:%s, rgb_image:%s, phog:%s]" % fileName, rgb_image, phog
        return string
    def __repr__(self):
        return str(self)

def getFiles():
    files = {}
    for fileName in os.listdir(DATA_DIR_PATH):
        if VERBOSE:
            print fileName

        fileId, fileExtension = fileName.split('.')

        fileId = fileId[:-2] # removes _c, _d
        if fileId not in files:
            files[fileId] = {}
        filePath = os.path.join(DATA_DIR_PATH, fileName)
        try:
            files[fileId][fileExtension] = scipy.misc.imread(filePath)
        except:
            strData = np.array(open(filePath).read().split())
            files[fileId][fileExtension] = strData.astype(np.float)

    print files
    return files

def retreiveKTrainingImages(k):
    # store the phogs, id -> phog
    # id should be same as files, id -> file

    dist = numpy.sum(numpy.subtract())

def main(image):
    files = getFiles()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: analogies.py [file/to/analogize]"
    else:
        main(sys.argv[1])
    main()

