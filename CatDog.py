import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def main():
    filestrain = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in os.listdir(r'../dogs-vs-cats/train/')]

    filestest = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in os.listdir(r'../dogs-vs-cats/test1/')]

    X_train = []
    X_test = []

    for i in range(filestrain.__sizeof__()):
        i = cv.resize(i, (128, 128))
        hog = cv.HOGDescriptor((128, 128), )

if __name__ == '__main__':
    main()


