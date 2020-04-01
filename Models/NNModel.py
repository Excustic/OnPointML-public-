import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
import time
import datetime as dt
from data_extractor import filename, transfrom_TimePoint
import pickle
import pygeohash as gh
import sys

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def train_model(path):
    data = pd.read_csv(filename, sep=",")
    centroids = pd.read_csv("../HDBSCAN_CLUSTER_CENTROIDS.csv").to_numpy()

    X_Time = data[
        ["year", "day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "season"]].to_numpy()
    y = data[["label"]].to_numpy()
    le = preprocessing.LabelEncoder()
    # cls = le.fit_transform(list(y))
    # y = np.array(list(cls))

    x_train, x_test, y_train, y_test = train_test_split(X_Time, y, test_size=0.1)
    print(type(x_train), type(X_Time))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(len(x_train) == len(y_train))

    input_layer_size = X_Time.shape[1]
    print()
    output_layer_size = len(list(set(y.flatten())))
    dense_layer_size = int(np.mean([input_layer_size, output_layer_size]))
    print("input_size: ", input_layer_size, "\nout_size: ", output_layer_size, "\ndense_size: ", dense_layer_size)

    # Model training and saving

    model = keras.Sequential([
        keras.layers.Input(shape=(input_layer_size,)),
        keras.layers.Dense(dense_layer_size, activation="relu"),
        keras.layers.Dense(output_layer_size, activation="softmax")
    ])
    max_acc = 0.0
    for i in range(4):
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=5)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(i, "|accuracy: ", test_acc)

        prediction = model.predict(x_test)
        if test_acc > max_acc:
            max_acc = test_acc
            model.save(path/"NNMODEL", save_format='tf')
    try:
        with open('accuracies.csv', "wb") as f2:
            df = pd.read_csv(f2, sep=",")
            df['NN'] = max_acc
            df.to_csv(path/'accuracies.csv')
    except:
        d = [{'NN': max_acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(path/'accuracies.csv', index=None)

def predict_model(path, timestamp):
    # Opening a saved model
    centroids = pd.read_csv("../HDBSCAN_CLUSTER_CENTROIDS.csv").to_numpy()

    model = keras.models.load_model(path/"NNMODEL")

    model.predict(transfrom_TimePoint(timestamp))

