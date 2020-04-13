import sys
from os.path import join
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM
from sklearn import preprocessing
import numpy as np
from data_extractor import file_extracted_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def train_model():
    centroids = pd.read_csv(join("D:\\Work\\PythonML\\OnPoint\\recycle_bin\\HDBSCAN_CLUSTER_CENTROIDS.csv"), sep =",", header=None).to_numpy()

    data = pd.read_csv("D:\\Work\\PythonML\\OnPoint\\recycle_bin\\TransformedData.csv", sep=",")

    X_Time = data[
        ["day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "quarter"]].to_numpy()
    y = data[["label"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X_Time, y, test_size=0.1)


    model = Sequential()

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    # y_train = y_train.reshape(y_train.shape[0])
    # y_test = y_test.reshape(y_test.shape[0])


    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[0], activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))


def predict_model(path, timestamp):
    centroids = pd.read_csv("HDBSCAN_CLUSTER_CENTROIDS.csv").to_numpy()

    print(path, timestamp)

# train_model()