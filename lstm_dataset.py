import numpy as np
import scipy.io
import tensorflow as tf
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Read the data and convert dictionary to Pandas DataFrame
df = pd.read_excel(
    "C:/Users/hecto/Documents/Master/tfm/tfm-renewable-energy-deep-learning/2015_30min.xlsx",
    header=None)[0]

# Data preprocessing: each row will contain the 10 measures for each day , and the 10 measures for the following day
X = pd.DataFrame(np.array(df).reshape(-1, 10), columns=["col_{}".format(i) for i in range(0, 10)])

y = pd.DataFrame.copy(X)

y.columns = ["col_{}".format(i) for i in range(11, 21)]
y = y.drop(0)
y = y.reset_index(drop=True)
y.loc[len(y)] = np.zeros(10)

dfPreproccessed = pd.concat([X, y], axis=1)

print("DataFrame Preproccessed:")
print(dfPreproccessed)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# Model definition
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation="sigmoid"))

# Model compilation
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("X_TRAIN")
print(X_train.shape[1])

# Model training with GPU acceleration - TO DO: CHANGE EPOCHS TO 10
with tf.device("/GPU:0"):
    #https://stackoverflow.com/questions/42240376/dataframe-object-has-no-attribute-reshape
    history = model.fit(
        X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1)),
        y_train,
        epochs=1,
        batch_size=16,
        validation_data=(X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1)), y_val))

# Training and Validation loss curves
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Model evaluation with validation data
score = model.evaluate(X_val.reshape((X_val.shape[0], X_val.shape[1], 1)), y_val)
print('Validation loss:', score)