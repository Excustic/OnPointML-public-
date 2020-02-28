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
import pickle
import pygeohash as gh
import sys


def circular_hour(hour, minute):
    return np.sin(float(hour + minute / 60) * 15 * np.pi / 180), np.cos(
        float(hour + minute / 60) * 15 * np.pi / 180)


def circular_month(month):
    return np.sin(month * 30 * np.pi / 180), np.cos(month * 30 * np.pi / 180)


def transfrom_TimePoint(Time):
    current_date = dt.datetime.utcfromtimestamp(Time / 1000).strftime("%Y/%m/%d %H:%M")
    yymmdd = current_date.split(" ")[0]
    hhmm = current_date.split(" ")[1]
    year = int(yymmdd.split("/")[0])
    month = int(yymmdd.split("/")[1])
    (year, week_of_year, day_of_week) = dt.datetime.isoweekday(
        dt.datetime.utcfromtimestamp(Time / 1000))  # monday - 1 , sunday - 7
    hour = int(hhmm.split(":")[0])
    minute = int(hhmm.split(":")[1])
    (hoursin, hourcos) = circular_hour(hour, minute)
    (monthsin, monthcos) = circular_month(month)
    is_weekend = 1 if (day_of_week == 5 or day_of_week == 6) else 0
    season = (np.ceil(month / 4))
    print(year, day_of_week, month, hoursin, hourcos, monthsin, monthcos, is_weekend, int(season))
    return year, day_of_week, hoursin, hourcos, monthsin, monthcos, is_weekend, season, week_of_year


config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

data = pd.read_csv("TransformedData.csv", sep=",", nrows=30000)

X_Time = data[
    ["year", "day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "season"]].to_numpy()
y = data[["result"]].to_numpy()
le = preprocessing.LabelEncoder()
cls = le.fit_transform(list(y))
y = np.array(list(cls))

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

# model = keras.Sequential([
#     keras.layers.Input(shape=(input_layer_size,)),
#     keras.layers.Dense(dense_layer_size, activation="relu"),
#     keras.layers.Dense(output_layer_size, activation="softmax")
# ])
# max_acc = 0.0
# for i in range(10):
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#     model.fit(x_train, y_train, epochs=5)
#
#     test_loss, test_acc = model.evaluate(x_test, y_test)
#     print(i, "|accuracy: ", test_acc)
#
#     prediction = model.predict(x_test)
#     if test_acc > max_acc:
#         max_acc = test_acc
#         model.save("NNMODEL", save_format='tf')

# Opening a saved model

model = keras.models.load_model("NNMODEL")

test_loss, test_acc = model.evaluate(x_test, y_test)
print("loss: ", test_loss, " acc: ", test_acc)
