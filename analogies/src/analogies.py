import numpy as np
import os
from scipy import ndimage, misc
import sys
import time
from phog_features.phog import PHogFeatures
from CLPatchMatch.main import CLPatchMatch
import operator
import matplotlib.pyplot as plt


VERBOSE = False

DATA_DIR_PATH = '../data_2'

phog = PHogFeatures()
patchmatch = CLPatchMatch()

class Image(object):

    def load_depth(self, dirPath, imageId):
        depthFileName = imageId + '_depthcrop.png'
        depthPath = os.path.join(dirPath, depthFileName)
        depthData = misc.imread(depthPath, mode='F')
        return depthData

    def load_rgb(self, dirPath, imageId):
        rgbFileName = imageId + '_crop.png'
        rgbPath = os.path.join(dirPath, rgbFileName)
        rgbData = misc.imread(rgbPath, mode='RGB')
        return rgbData

    def load_phog(self, dirPath, imageId):
        rgbFileName = imageId + '_crop.png'
        rgbPath = os.path.join(dirPath, rgbFileName)
        phogData = phog.get_features(rgbPath)
        startIndex = (len(phogData) - 680)/2
        endIndex = startIndex + 680
        return phogData[startIndex:endIndex]

    def __init__(self, dirPath, imageId):
        self.id = imageId
        self.rgbPath = os.path.abspath(os.path.join(dirPath, imageId + '_crop.png'))
        self.depthPath = os.path.abspath(os.path.join(dirPath, imageId + '_depthcrop.png'))
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
    lastUnderscoreIndex = imageId.rfind('_')
    imageId = imageId[:lastUnderscoreIndex]
    return imageId

def load_training_images(dirPath):
    fileNames = os.listdir(dirPath)
    imageIds = {get_image_id(fileName) for fileName in fileNames}
    images = {imageId:Image(dirPath, imageId) for imageId in imageIds}
    return images

def infer_depth_gradient(pixelRGB, pixelRow, pixelCol, kImages):
    dists = []
    depthGradientSamples = {}
    for k, image in enumerate(kImages):
        # print image.patchmatch.shape, pixelRow, pixelCol
        effectiveRow = np.clip(pixelRow, 0, len(image.patchmatch)-1)
        effectiveCol = np.clip(pixelCol, 0, len(image.patchmatch[0])-1)
        matchCoords = image.patchmatch[effectiveRow][effectiveCol]
        matchRow, matchCol = int(matchCoords[1]), int(matchCoords[0])
        # print "offset " + str(offset)
        # print image.depthGradient
        print image.id, image.patchmatch.shape, pixelRow, pixelCol, effectiveRow, effectiveCol, matchRow, matchCol, image.depthGradient[0].shape
        depthGradientSamples[k] = (image.depthGradient[0][matchRow][matchCol], image.depthGradient[1][matchRow][matchCol])
        rgbSample = image.rgb[matchRow][matchCol]
        dist = np.sum((pixelRGB - rgbSample) * (pixelRGB - rgbSample))
        dists.append(dist)
    dists = np.array(dists)
    distSum = np.sum(dists)
    kWeights = 1.0 - np.divide(dists, distSum)
    weightSum = np.sum(kWeights)*0.5
    weightAccumulator = 0
    kStar = 0
    for k, weight in enumerate(kWeights):
        weightAccumulator += weight
        if weightAccumulator >= weightSum:
            kStar = k
            break
    return depthGradientSamples[kStar]

def main(inputPath):
    trainingImages = load_training_images(DATA_DIR_PATH)
    kImages = retreive_k_training_images(inputPath, trainingImages, 7)
    print [image.id for image in kImages]
    inputRGB = misc.imread(inputPath, mode='RGB')
    for image in kImages:
        image.patchmatch = patchmatch.match([image.rgbPath, inputPath])
        # plt.figure()
        # plt.imshow(image.patchmatch)
        # plt.show()
        # print image.patchmatch
    inputDepthGradient = np.zeros(inputRGB.shape)
    for row in range(len(inputRGB)):
        for col, pixel in enumerate(inputRGB[row]):
            inputDepthGradient[row][col][0], inputDepthGradient[row][col][1] = infer_depth_gradient(pixel, row, col, kImages)
    print inputDepthGradient.shape
    plt.figure()
    plt.imshow(inputDepthGradient)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: analogies.py [file/to/analogize]"
    else:
        inputPath = os.path.abspath(sys.argv[1])
        main(inputPath)
