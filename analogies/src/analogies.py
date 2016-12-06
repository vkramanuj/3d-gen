import numpy as np
import os
import cv2
from scipy import ndimage, misc
import sys
import time
from phog_features.phog import PHogFeatures
from patchmatch.main import CLPatchMatch
import operator
import matplotlib.pyplot as plt


VERBOSE = False

DATA_DIR_PATH = '../data_3'

phog = PHogFeatures()
patchmatch = CLPatchMatch()
# sift = cv2.SIFT()
# dense = cv2.FeatureDetector_create("Dense")
# dense.setInt('initXyStep', 1)

class Image(object):

    def load_depth(self, dirPath, imageId):
        depthData = misc.imread(self.depthPath, mode='F')
        depthData = misc.imresize(depthData, self.targetSize)
        return depthData

    def load_rgb(self, dirPath, imageId):
        rgbData = misc.imread(self.rgbPath, mode='RGB')
        rgbData = misc.imresize(rgbData, self.targetSize)
        return rgbData

    def load_phog(self, dirPath, imageId):
        phogData = phog.get_features(self.rgbPath)
        startIndex = (len(phogData) - 680)/2
        endIndex = startIndex + 680
        return phogData[startIndex:endIndex]

    def load_sift(self, dirPath, imageId):
        # img = cv2.imread(self.rgbPath)
        imgGray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
        kp = dense.detect(imgGray)
        kp, des = sift.compute(imgGray, kp)
        pixelDescriptors = des.reshape((self.targetSize[0], self.targetSize[1], 128)).astype(np.uint8)
        return pixelDescriptors
        # sifts = []
        # for row in range(len(self.rgb)):
        #     for col in range(len(self.rgb[row])):
        # return np.array(sifts)

    def __init__(self, targetSize, dirPath, imageId):
        self.targetSize = targetSize
        self.dirPath = dirPath
        self.id = imageId
        # self.rgbPath = os.path.abspath(os.path.join(dirPath, imageId + '_crop.png'))
        self.rgbPath = os.path.abspath(os.path.join(dirPath, imageId + '-color.png'))
        # self.depthPath = os.path.abspath(os.path.join(dirPath, imageId + '_depthcrop.png'))
        self.depthPath = os.path.abspath(os.path.join(dirPath, imageId + '-depth.png'))
        self.rgb = self.load_rgb(dirPath, imageId)
        self.phog = self.load_phog(dirPath, imageId)

    def process_as_k_image(self):
        self.depth = self.load_depth(self.dirPath, self.id)
        self.depthGradient = np.gradient(self.depth)
        # self.sift = self.load_sift(self.dirPath, self.id)

    def __str__(self):
        string = "IMAGE[id:%s, rgb:%s, phog:%s, depth:%s]" % (self.id, self.rgb, self.phog, self.depth)
        return string

    def __repr__(self):
        return str(self)

def retreive_k_training_images(inputImage, trainingImages, k):
    inputPhog = phog.get_features(inputPath)
    startIndex = (len(inputPhog) - 680)/2
    endIndex = startIndex + 680
    inputPhog = inputPhog[startIndex:endIndex]
    dists = []
    for _, trainingImage in trainingImages.items():
        dist = np.sum(np.power(np.subtract(inputPhog, trainingImage.phog), 2))
        dists.append((trainingImage, dist))
    sortedDists = sorted(dists, key=operator.itemgetter(1))
    kClosestImages = [dist[0] for dist in sortedDists[:k]]
    return kClosestImages

def get_image_id(fileName):
    imageId = fileName.split('.')[0]
    # lastUnderscoreIndex = imageId.rfind('_')
    lastUnderscoreIndex = imageId.rfind('-')
    imageId = imageId[:lastUnderscoreIndex]
    return imageId

def load_training_images(inputId, targetSize, dirPath):
    fileNames = os.listdir(dirPath)
    imageIds = {get_image_id(fileName) for fileName in fileNames}
    images = {imageId:Image(targetSize, dirPath, imageId) for imageId in imageIds if not imageId == inputId}
    return images

def sample_depth_gradient(image, pixelRow, pixelCol):
    matchCoords = image.patchmatch[pixelRow][pixelCol]
    # print image.id, pixelRow, pixelCol, matchCoords
    matchRow = np.clip(int(matchCoords[0]), 0, len(image.depthGradient[0])-1)
    matchCol = np.clip(int(matchCoords[1]), 0, len(image.depthGradient[0][0])-1)
    depthGradientSample = (
        image.depthGradient[0][matchRow][matchCol], image.depthGradient[1][matchRow][matchCol])
    return depthGradientSample

def get_dist(pixelRGB, image, pixelRow, pixelCol):
    matchCoords = image.patchmatch[pixelRow][pixelCol]
    # print image.id, pixelRow, pixelCol, matchCoords
    matchRow = np.clip(int(matchCoords[0]), 0, len(image.depthGradient[0])-1)
    matchCol = np.clip(int(matchCoords[1]), 0, len(image.depthGradient[0][0])-1)
    rgbSample = image.rgb[matchRow][matchCol]
    dist = float(np.sum((pixelRGB - rgbSample) * (pixelRGB - rgbSample)))
    return dist

def infer_depth_gradient(pixelRGB, pixelRow, pixelCol, kImages):
    depthGradientSamples = np.array(
        [sample_depth_gradient(image, pixelRow, pixelCol) for image in kImages])
    dists = np.array(
        [get_dist(pixelRGB, image, pixelRow, pixelCol) for image in kImages])
    distSum = np.sum(dists)
    if distSum == 0.0 or len(dists) == 1:
        kWeights = np.ones(dists.shape)/len(dists)
    else:
        kWeights = 1.0 - np.divide(dists, distSum)
    # return np.matmul(kWeights, np.array(depthGradientSamples))
    weightSum = np.sum(kWeights)*0.5
    weightAccumulator = 0
    kStar = 0
    for k, weight in sorted(enumerate(kWeights), key=operator.itemgetter(1)):
        weightAccumulator += weight
        if weightAccumulator >= weightSum:
            kStar = k
            break
    return depthGradientSamples[kStar]

def main(inputPath):
    inputRGB = misc.imread(inputPath, mode='RGB')
    workingSize = (300, 400)
    inputRGB = misc.imresize(inputRGB, workingSize)
    # inputGray = cv2.cvtColor(inputRGB, cv2.COLOR_BGR2GRAY)
    # keypoints = dense.detect(inputGray)
    # descriptors = sift.compute(inputGray, keypoints)
    # pixelDescriptors = descriptors[1].reshape((workingSize[0], workingSize[1], 128)).astype(np.uint8)
    # inputSift = pixelDescriptors
    inputId = get_image_id(os.path.basename(inputPath))
    trainingImages = load_training_images(inputId, inputRGB.shape[0:2], DATA_DIR_PATH)
    kImages = retreive_k_training_images(inputPath, trainingImages, 3)
    print [image.id for image in kImages]
    for image in kImages:
        image.process_as_k_image()
        image.patchmatch = patchmatch.match([inputRGB, image.rgb], Demo=True)
        # plt.figure()
        # plt.imshow(image.patchmatch)
        # plt.show()
        print image.id
    inputDepthGradient = np.zeros((workingSize[0], workingSize[1], 2), np.float)
    for row in range(len(inputRGB)):
        for col, pixel in enumerate(inputRGB[row]):
            inputDepthGradient[row][col] = infer_depth_gradient(pixel, row, col, kImages)
    inputDepthGradient[:][:][0] = cv2.bilateralFilter(inputDepthGradient[:][:][0].astype(np.float32),5,75,75)
    inputDepthGradient[:][:][1] = cv2.bilateralFilter(inputDepthGradient[:][:][1].astype(np.float32),5,75,75)
    inputDepthGradient = np.dstack([inputDepthGradient, np.zeros(workingSize)])
    plt.figure()
    plt.imshow(inputDepthGradient)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: analogies.py [file/to/analogize]"
    else:
        inputPath = os.path.abspath(sys.argv[1])
        main(inputPath)
