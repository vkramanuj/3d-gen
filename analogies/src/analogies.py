import numpy as np
import os
from scipy import ndimage, misc
import sys
from phog_features.phog import PHogFeatures
from CLPatchMatch.main import CLPatchMatch
import operator

VERBOSE = False

DATA_DIR_PATH = '../data'

phog = PHogFeatures()
patchmatch = CLPatchMatch()

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

    def __init__(self, dirPath, imageId):
        self.id = imageId
        self.rgbPath = os.path.abspath(os.path.join(dirPath, imageId + '_c.bmp'))
        self.depthPath = os.path.abspath(os.path.join(dirPath, imageId + '_d.dat'))
        self.rgb = self.load_rgb(dirPath, imageId)
        self.phog = self.load_phog(dirPath, imageId)
        self.depth = self.load_depth(dirPath, imageId)
        self.depthGradient = np.gradient(self.depth)

    def __str__(self):
        string = "IMAGE[id:%s, rgb:%s, phog:%s, depth:%s]" % (self.id, self.rgb, self.phog, self.depth)
        return string

    def __repr__(self):
        return str(self)

def retreive_k_training_images(inputImage, trainingImages, k):
    inputPhog = phog.get_features(inputPath)
    dists = []
    for _, trainingImage in trainingImages.items():
        dist = np.sum(np.power(np.subtract(inputPhog, trainingImage.phog), 2))
        dists.append((trainingImage, dist))
    sortedDists = sorted(dists, key=operator.itemgetter(1))
    kClosestImages = [dist[0] for dist in sortedDists[:k]]
    return kClosestImages

def get_image_id(fileName):
    imageId = fileName.split('.')[0]
    imageId = imageId[:-2] # removes _c, _d
    return imageId

def load_training_images(dirPath):
    fileNames = os.listdir(dirPath)
    imageIds = {get_image_id(fileName) for fileName in fileNames}
    images = {imageId:Image(dirPath, imageId) for imageId in imageIds}
    return images

def main(inputPath):
    trainingImages = load_training_images(DATA_DIR_PATH)
    kImages = retreive_k_training_images(inputPath, trainingImages, 7)
    for image in kImages:
        image.patchmatch = patchmatch.match(inputPath, image.rgbPath)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: analogies.py [file/to/analogize]"
    else:
        inputPath = sys.argv[1]
        main(inputPath)
