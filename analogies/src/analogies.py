import numpy as np
import os
from scipy import ndimage, misc
import sys
from phog_features.phog import PHogFeatures

VERBOSE = False

DATA_DIR_PATH = '../data'

phog = PHogFeatures()

trainingImages = {}
inputImage = None

class Image(object):

    def load_depth(self, dirPath, imageId):
        depthFileName = imageId + '_d.dat'
        depthPath = os.path.join(dirPath, depthFileName)
        strDepthData = np.array(open(depthPath).read().split())
        depthData = strDepthData.astype(np.int)
        return depthData

    def load_rgb(self, dirPath, imageId):
        rgbFileName = imageId + '_c.bmp'
        rgbPath = os.path.join(dirPath, rgbFileName)
        rgbData = misc.imread(rgbPath, mode='RGB')
        return rgbData

    def load_phog(self, dirPath, imageId):
        rgbFileName = imageId + '_c.bmp'
        rgbPath = os.path.join(dirPath, rgbFileName)
        phogData = phog.get_features(rgbPath)
        return phogData

    def __init__(self, imageId):
        self.id = imageId
        self.rgb = self.load_rgb(DATA_DIR_PATH, imageId)
        self.phog = self.load_phog(DATA_DIR_PATH, imageId)
        self.depth = self.load_depth(DATA_DIR_PATH, imageId)
        print self

    def __str__(self):
        string = "IMAGE[id:%s, rgb:%s, phog:%s, depth:%s]" % (self.id, self.rgb, self.phog, self.depth)
        return string

    def __repr__(self):
        return str(self)

def retreive_k_training_images(k):
    # store the phogs, id -> phog
    # id should be same as files, id -> file
    dist = numpy.sum(numpy.subtract())

def get_image_id(fileName):
    imageId = fileName.split('.')[0]
    imageId = imageId[:-2] # removes _c, _d
    return imageId

def load_training_images():
    fileNames = os.listdir(DATA_DIR_PATH)
    imageIds = {get_image_id(fileName) for fileName in fileNames}
    images = {imageId:Image(imageId) for imageId in imageIds}
    return images

def main(image):
    trainingImages = load_training_images()
    inputImage = image
    # print images

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: analogies.py [file/to/analogize]"
    else:
        main(sys.argv[1])
    main()
