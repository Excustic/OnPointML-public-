import os
from os.path import join
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_extractor import file_extracted_data, extract_single, file_cluster_centroids, file_accuracies
import sys

folder_name = "NNMODEL"
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def train_model(path):
    # preparing data for model

    data = pd.read_csv(os.path.join(sys.path[0], path, file_extracted_data), sep=",")
    centroids = pd.read_csv(os.path.join(path, file_cluster_centroids)).to_numpy()

    X_Time = data[
        ["day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "quarter"]].to_numpy()
    y = data[["label"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X_Time, y, test_size=0.1)
    print(type(x_train), type(X_Time))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(len(x_train) == len(y_train))

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
    for i in range(4):
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=5)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(i, "|accuracy: ", test_acc)

        if test_acc > max_acc:
            max_acc = test_acc
            model.save(join(path, folder_name), save_format='tf')
    try:
        with open(join(sys.path[0], path, file_accuracies), "wb") as f2:
            df = pd.read_csv(f2, sep=",")
            df['NN'] = max_acc
            df.to_csv(join(path, file_accuracies))
    except:
        d = [{'NN': max_acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(join(path, file_accuracies), index=False)


def predict_model(path, timestamp):
    # Opening a saved model
    centroids = pd.read_csv(join(sys.path[0], path, file_cluster_centroids)).to_numpy()

    model = keras.models.load_model(join(sys.path[0], path, folder_name))

    p = model.predict(extract_single(timestamp))

    points = sorted(
        [{'latitude': lat, 'longitude': long, 'radius': r, 'confidence': conf} for lat, long, r, conf in zip(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], p.reshape(p.shape[1]))], key=lambda x: x['confidence'],
        reverse=True)

    return points
