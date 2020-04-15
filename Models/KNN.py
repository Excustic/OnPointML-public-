#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, OnPoint Project"
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import json
import numpy as np
import sys
from os.path import join

import pytz
from tzwhere import tzwhere
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from data_extractor import file_extracted_data, extract_single, file_cluster_centroids, file_accuracies, \
    file_home_cluster
import pickle

# default naming for model
Model_file_name = "KNNMODEL.pickle"


# training a model that uses K-Nearest Neighbors algorithm
def train_model(path):
    # preparing data for model
    min_neighbors, max_neighbors = 5, 23
    data = pd.read_csv(join(path, file_extracted_data), sep=",")
    centroids = pd.read_csv(join(sys.path[0], path, file_cluster_centroids), sep=",", header=None).to_numpy()

    x_time = data[
        ["day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "quarter"]].to_numpy()
    y = data[["label"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x_time, y, test_size=0.1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Model training a saving
    max_acc = 0.0
    for count in range(min_neighbors, max_neighbors, step=2):  # K-Neighbors requires an odd number
        model = KNeighborsClassifier(n_neighbors=count)

        model.fit(x_train, y_train)
        test_acc = model.score(x_test, y_test)

        if test_acc > max_acc:
            max_acc = test_acc
            with open(join(sys.path[0], path, Model_file_name), "wb") as f:
                pickle.dump(model, f)

    # save best model and its accuracy to later on determine the optimal model
    try:
        with open(join(sys.path[0], path, file_accuracies), "wb") as f2:
            df = pd.read_csv(f2)
            df['KNN'] = max_acc
            df.to_csv(file_accuracies)
    except:
        d = [{'KNN': max_acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(join(path, file_accuracies), index=False)


# perform a prediction on a timestamp using the trained model
def predict_model(path, timestamp):
    # preparing data and importing model
    centroids = pd.read_csv(join(sys.path[0], path, file_cluster_centroids), sep=",", header=None).to_numpy()
    model = pickle.load(open(join(sys.path[0], path, Model_file_name), 'rb'))

    # determine user's timezone
    home = pd.read_csv(join(sys.path[0], path, file_home_cluster), sep=',', header=None).to_numpy()
    tz = tzwhere.tzwhere()
    timezone_str = pytz.timezone(tz.tzNameAt(home[0], home[1]))

    # predict and receive list of confidences
    result = model.predict_proba(extract_single(timestamp, timezone_str))

    points = sorted(
        [{'latitude': str(lat), 'longitude': str(long), 'radius': str(r), 'confidence': str(conf)} for
         lat, long, r, conf in zip(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], result.reshape(result.shape[1]))],
        key=lambda x: x['confidence'],
        reverse=True)

    return points
