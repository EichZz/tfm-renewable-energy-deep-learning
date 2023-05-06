import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Convert dictionary to Pandas DataFrame
df = pd.read_excel('2015_30min.xlsx', header= None)[0]

# Data preprocessing: each row will contain the 20 day measures, and the 20 measures 
# who will be predicted for the next day by the algorithm.
print("Num rows: " + str(df.shape[0]))
numRows = int(df.shape[0] / 10)

dfMulti = pd.DataFrame(columns=['col' + str(i) for i in range(0, 9)])

for i in range(numRows):
    start = 10 * i
    end = start + 10
    print("Start: " + str(start) + ". End: " + str(end) + ". " + str(df.iloc[start:end]))
    dfMulti.loc[i] = df.iloc[start:end].values.flatten()

print(dfMulti)


# target = 
# define the model
# model = Sequential()
# model.add(LSTM(32, input_shape=(10, 1)))
# model.add(Dense(1, activation='sigmoid'))

# # compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model with GPU acceleration
# with tf.device('/GPU:0'):
#   model.fit(data, target, epochs=10, batch_size=32)


# In this example, we first generate some sample data where each sample is a sequence of 10 time steps with one feature. 
# We then define an LSTM model with 32 units and a dense layer with a sigmoid activation function.
# We compile the model using binary cross-entropy loss and the Adam optimizer, and then train the model on the sample data for 10 epochs with a batch size of 32.
# The with tf.device('/GPU:0'): statement specifies that the model training should be run on the first available GPU. 
# This ensures that the training is accelerated using GPU hardware.