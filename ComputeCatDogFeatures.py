import numpy as np
import cv2 as cv
import glob

data_folder_train = 'train/*.jpg'
data_folder_test = 'test1/*.jpg'

winSize = (200, 200)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

winSize=(8, 8)
padding=(8, 8)

def main():
    filestrain = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.iglob(data_folder_train)]
    filestest = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.iglob(data_folder_test)]

    X_train = []
    X_test = []

    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    for i in filestrain:
        i = cv.resize(i, winSize)
        hog.compute(i, winSize, padding)

    for i in filestest:
        i = cv.resize(i, winSize)
        hog.compute(i, winSize, padding)


if __name__ == '__main__':
    main()


