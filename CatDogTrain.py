import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

csv = 'train_4_features.csv'

def prepareDataframe(frame):
    frame["Filename"] = frame["Filename"].replace(to_replace=r'^cat.*', value='-1', regex=True)
    frame["Filename"] = frame["Filename"].replace(to_replace=r'^dog.*', value='1', regex=True)
    frame["Filename"] = frame["Filename"].astype(float)

    dataY = frame["Filename"].values

    data = frame.drop(["Filename"], axis=1)
    dataX = data.values

    return dataX, dataY

def buildModel(inputShape):
    model = Sequential()
    model.add(Input(shape=inputShape))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.compile(optimizer='adam',
                  loss='huber_loss',
                  metrics=['accuracy'])

    batchSize = 128
    epochs = 30

    return model, batchSize, epochs

def getMetrics(pred, ground):
    acc = accuracy_score(y_true=ground, y_pred=pred)
    return acc

def main():
    trainFrame = pd.read_csv(csv)
    trainX, trainY = prepareDataframe(trainFrame)
    trainX.reshape((1, 4000, 576))

#################ADA BOOST CLASSIFIER################################
    classifier = AdaBoostClassifier(n_estimators=100)
    classifier.fit(trainX, trainY)
    predTrain = classifier.predict(trainX)
    accTrain = getMetrics(predTrain, trainY)

    print("Training accuracy: " + accTrain.astype(str))
#####################################################################

####################KERAS MODEL######################################
    model, batch_size, epochs = buildModel((576, ))
    model.summary()
    model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs)
#####################################################################

if __name__ == '__main__':
    main()