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
import sys
from os.path import join
import keras
import pandas as pd
import pytz
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tzwhere import tzwhere
from data_extractor import file_extracted_data, file_cluster_centroids, file_accuracies, file_home_cluster, \
    extract_single
from sklearn.model_selection import train_test_split

# default naming for model
folder_name = "DEEPMODEL"

# configurations for the usage gpu_tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# training a Deep Learning model that uses Long Short Term Memory cells
def train_model(path):
    print(folder_name)
    sessions_count = 1
    epochs = 5
    data = pd.read_csv(os.path.join(sys.path[0], path, file_extracted_data), sep=",")
    x_time = data[
        ["day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "quarter"]].to_numpy()
    y = data[["label"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x_time, y, test_size=0.1)

    output_layer_size = len(list(set(y.flatten())))

    model = Sequential()

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(output_layer_size, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    max_acc = 0.0
    for i in range(sessions_count):
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(i, "|accuracy: ", test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            model.save(join(path, folder_name), save_format='tf')
    # save best model and its accuracy to later on determine the optimal model
    try:
        df = pd.read_csv(join(sys.path[0], path, file_accuracies))
        df['DEEP'] = max_acc
        df.to_csv(join(sys.path[0], path, file_accuracies), index=False)
    except EnvironmentError:
        d = [{'DEEP': max_acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(join(path, file_accuracies), index=False)
    return True


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
        [{'latitude': str(lat), 'longitude': str(long), 'radius': str(r), 'confidence': str(conf)} for
         lat, long, r, conf in zip(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], p.reshape(p.shape[1]))], key=lambda x: x['confidence'],
        reverse=True)

    return points
