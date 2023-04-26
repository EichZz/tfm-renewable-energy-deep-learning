import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dropout
import tensorflow_addons as tf_addons
from tensorflow_addons.layers import TimeDistributedTCN


# Read the data from the matlab file
mat = scipy.io.loadmat('PV1min_Cmb.mat')

# Convert dictionary to Pandas DataFrame
df = pd.DataFrame(mat.items())
print(df)
values = df.values[3]

np.savetxt('dataset.txt', values, delimiter=',')

#target = 

# define the model
model = Sequential()
model.add(TimeDistributedTCN(filters=64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# train the model with GPU acceleration
# with tf.device('/GPU:0'):
#   model.fit(data, target, epochs=10, batch_size=32)

# In this example, we first generate some sample data where each sample is a sequence of 10 time steps with one feature. 
# We then define a TCN model with 64 filters and a kernel size of 3, followed by a flatten layer and a dense layer with a sigmoid activation function. 
# We compile the model using binary cross-entropy loss and the Adam optimizer, and then train the model on the sample data for 10 epochs with a batch size of 32. 
# The with tf.device('/GPU:0'): statement specifies that the model training should be run on the first available GPU. This ensures that the training is accelerated using GPU hardware.
# Note that in this example we are using the TimeDistributedTCN layer from the TensorFlow Addons library, which allows us to apply the TCN to each time step of the input sequence independently.
# This is useful for many sequence prediction tasks.