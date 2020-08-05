import os
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from time import time
from tensorflow.keras.callbacks import TensorBoard


mode = 'ratings' # ratings or points

def get_sof(ratings):
    sum_ratings = 0
    for rating in ratings:
        sum_ratings += math.exp(-rating / (1600 / math.log(2)))
    return int((1600 / math.log(2)) * math.log(len(ratings) / sum_ratings))

class Mask(keras.layers.Layer):
  def call(self, inputs):
    return tf.where(inputs[1] == 0, inputs[1], inputs[0])

# runs through input data folder and forms training/validate sets
# data should be clear of unofficial or rookie races
def getTrainingData(fileNames, dataPath):
    X = np.zeros((len(fileNames), 64))
    y = np.zeros((len(fileNames), 64))
    i = 0
    for f in fileNames:
        session = pd.read_csv(dataPath + f, header=0, delimiter=';', encoding='latin1')
        ratings = session[['rating_old']].to_numpy()
        ratingChanges = session[['rating_delta']].to_numpy()
        points = session[['points']].to_numpy()
            
        if mode == 'ratings':
            sof = get_sof(ratings)
            ratings = ratings / sof
            ratingChanges[ratingChanges != 0] += 100
            ratingChanges = ratingChanges / 200
            for j in range(ratingChanges.shape[0]):
                y[i][j] = ratingChanges[j]

        if mode == 'points':
            for j in range(points.shape[0]):
                y[i][j] = points[j]

        for j in range(ratings.shape[0]):
            X[i][j] = ratings[j]

        i += 1
    
    return X, y



trainingDataPath = str('D:/Programs/python/ir_ratings_tf/data/')
validateDataPath = str('D:/Programs/python/ir_ratings_tf/data_validate/')

fileNames = [f for f in listdir(trainingDataPath) if isfile(join(trainingDataPath, f))]
validationFileNames = [f for f in listdir(validateDataPath) if isfile(join(validateDataPath, f))]


X, y = getTrainingData(fileNames, trainingDataPath)
X_val, y_val = getTrainingData(validationFileNames, validateDataPath)

# define model
inp = keras.layers.Input(shape=(64,))
hidden1 = keras.layers.Dense(256, activation='relu')(inp)
hidden2 = keras.layers.Dense(256, activation='relu')(hidden1)
out = keras.layers.Dense(64)(hidden2)
post = Mask(64)([out, inp])     # an additional custom layer that zeroes out trailing neurons

model = keras.Model(inputs=[inp], outputs=post)

model.compile(optimizer='adam',
              loss='mae')

# enables tensorboard metrics
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

model.fit(X, y, epochs=500, 
            validation_data=(X_val, y_val),
            callbacks=[tensorboard])


# test

y_pred = model.predict(X_val)

y_pred_flat = y_pred.reshape(-1, 1)
y_test_flat = y_val.reshape(-1, 1)

# unnormalize ratings data
if mode == 'ratings':
    y_pred_flat = y_pred_flat * 200
    y_pred_flat[y_pred_flat != 0] -= 100
    y_test_flat = y_test_flat * 200
    y_test_flat[y_test_flat != 0] -= 100

err = y_pred_flat - y_test_flat

# plotting
fig, axes = plt.subplots(ncols=2, figsize=(8, 8))
ax1, ax2 = axes
ax1.scatter(y_pred_flat, y_test_flat)
ax1.set_title('Actual vs predicted')
ax1.set_xlabel('Actual ' + mode)
ax1.set_ylabel('Predicted ' + mode)
ax2.hist(err, rwidth = 0.9, bins=20)
ax2.set_title('Error distribution')

plt.show()

# sample code to predict single session results

# for f in validationFileNames:
#     session = pd.read_csv(validateDataPath + f, header=0, delimiter=';', encoding='latin1')
#     ratings = session[['rating_old']].to_numpy()
#     ratingChanges = session[['rating_delta']].to_numpy()
#     points = session[['points']].to_numpy()

#     sof = get_sof(ratings)
#     if mode == 'ratings':
#         ratings = ratings / sof

#     X_test = np.zeros((64, 1))
#     X_test[:ratings.shape[0],:ratings.shape[1]] = ratings

#     y_test = np.zeros((64, 1))
#     if mode == 'ratings':
#         y_test[:ratingChanges.shape[0],:ratingChanges.shape[1]] = ratingChanges

#     if mode == 'points':
#         y_test[:points.shape[0],:points.shape[1]] = points
    

#     X_test = X_test.reshape(1, 64)
#     y_test = y_test.reshape(1, 64)
#     y = model.predict(X_test)
#     y = y.reshape(64, 1)
#     y_test = y_test.reshape(64, 1)
#     if mode == 'ratings':
#         y = y * 200
#         y[y != 0] -= 100
#     print(f, sof)
#     for i in range(64):
#         print(y[i][0], y_test[i][0], abs(y[i][0] - y_test[i][0]))
#     print("---")