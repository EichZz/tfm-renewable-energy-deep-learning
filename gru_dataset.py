import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, GRU


# Read the data from the matlab file
mat = scipy.io.loadmat('PV1min_Cmb.mat')

# Convert dictionary to Pandas DataFrame
df = pd.DataFrame(mat.items())
print(df)
values = df.values[3]

np.savetxt('dataset.txt', values, delimiter=',')

#target = np.random.randint(0, 2, (100, 1))

# define the model
model = Sequential()

# GRU layer with 32 units, Input sequences of any length, Dimensionality 1
model.add(GRU(32, input_shape=(None, 1)))

# Activation funtion --> Sigmoid
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model with GPU acceleration
#with tf.device('/GPU:0'):
# model.fit(df, target, epochs=10, batch_size=32)