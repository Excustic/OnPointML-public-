import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
import time
from data_extractor import filename, transfrom_TimePoint
import datetime as dt
import pickle
import pygeohash as gh
import logging

# # give unique ids for each geohash
# le = preprocessing.LabelEncoder()
# cls = le.fit_transform(list(geohash))
#
# X_Time = transfrom_X(Time)
# y = list(cls)
# # X_Time = preprocessing.normalize(X_Time)
# predict = ["geohash"]

def train_model(path):
    data = pd.read_csv(path/filename, sep=",")
    centroids = pd.read_csv(path+"HDBSCAN_CLUSTER_CENTROIDS.csv").to_numpy()

    X_Time = data[
        ["year", "day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "season"]].to_numpy()
    y = data[["label"]].to_numpy()
    le = preprocessing.LabelEncoder()
    cls = le.fit_transform(list(y))
    y = np.array(list(cls))

    x_train, x_test, y_train, y_test = train_test_split(X_Time, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=7)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    predicted = model.predict(x_test)
    # for i in range(len(predicted)):
    #     print("Predicted: ", le.classes_[predicted[i]], "Actual: ", le.classes_[y_test[i]], centroids[le.classes_[y_test[i]]-1])
    #
    # print(int(time.time()))
    #
    # print(le.inverse_transform(model.predict([transfrom_TimePoint(int(1576564333*1000))])))
    #
    # print(acc)

    with open("KNNModel.pickle", "wb") as f:
        pickle.dump(model, f)
    try:
        with open(path/'accuracies.csv', "wb") as f2:
            df = pd.read_csv(f2, sep=",")
            df['KNN'] = acc
            df.to_csv('accuracies.csv')
    except:
        d = [{'KNN': acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(path/'accuracies.csv', index=None)


def predict_model(path, timestamp):
    centroids = pd.read_csv("../HDBSCAN_CLUSTER_CENTROIDS.csv").to_numpy()

    model = pickle.load(open(path+"\\KNNModel.pickle", 'rb'))
    result = model.predict(transfrom_TimePoint(timestamp))
    return result
