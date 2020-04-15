#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, OnPoint Project"
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import os
from os.path import join
import pytz
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tzwhere import tzwhere

from data_extractor import file_extracted_data, extract_single, file_cluster_centroids, file_accuracies, \
    file_home_cluster
import sys

# default naming for model
folder_name = "NNMODEL"

# configurations for the usage gpu_tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# training a model that uses Neural Networks
def train_model(path):

    # preparing data for model
    sessions_count = 4
    epochs = 5
    data = pd.read_csv(os.path.join(sys.path[0], path, file_extracted_data), sep=",")
    centroids = pd.read_csv(os.path.join(path, file_cluster_centroids), sep=",", header=None).to_numpy()

    X_Time = data[
        ["day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "quarter"]].to_numpy()
    y = data[["label"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X_Time, y, test_size=0.1)
    print(type(x_train), type(X_Time))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    input_layer_size = X_Time.shape[1]
    print()
    output_layer_size = len(list(set(y.flatten())))
    dense_layer_size = int(np.mean([input_layer_size, output_layer_size]))

    # Model training and saving
    model = keras.Sequential([
        keras.layers.Input(shape=(input_layer_size,)),
        keras.layers.Dense(dense_layer_size, activation="relu"),
        keras.layers.Dense(output_layer_size, activation="softmax")
    ])
    max_acc = 0.0
    for i in range(sessions_count):
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=epochs)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(i, "|accuracy: ", test_acc)

        if test_acc > max_acc:
            max_acc = test_acc
            model.save(join(path, folder_name), save_format='tf')
    # save best model and its accuracy to later on determine the optimal model
    try:
        df = pd.read_csv(join(sys.path[0], path, file_accuracies))
        df['NN'] = max_acc
        df.to_csv(join(sys.path[0], path, file_accuracies), index=False)
    except:
        d = [{'NN': max_acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(join(path, file_accuracies), index=False)


def predict_model(path, timestamp):
    # preparing data and importing model
    centroids = pd.read_csv(join(sys.path[0], path, file_cluster_centroids), sep=",", header=None).to_numpy()
    model = keras.models.load_model(join(sys.path[0], path, folder_name))

    # determine user's timezone
    home = pd.read_csv(join(sys.path[0], path, file_home_cluster), sep=',', header=None).to_numpy()
    tz = tzwhere.tzwhere()
    timezone_str = pytz.timezone(tz.tzNameAt(home[0], home[1]))

    # predict and receive list of confidences
    p = model.predict(extract_single(timestamp, timezone_str))

    points = sorted(
        [{'latitude': str(lat), 'longitude': str(long), 'radius': str(r), 'confidence': str(conf)} for lat, long, r, conf in zip(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], p.reshape(p.shape[1]))], key=lambda x: x['confidence'],
        reverse=True)

    return points
