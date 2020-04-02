import json
import numpy as np
import sys
from os.path import join
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from data_extractor import file_extracted_data, extract_single, file_cluster_centroids, file_accuracies
import pickle

Model_file_name = "KNNMODEL.pickle"


def train_model(path):
    data = pd.read_csv(join(path, file_extracted_data), sep=",")
    centroids = pd.read_csv(join(sys.path[0], path, file_cluster_centroids)).to_numpy()

    x_time = data[
        ["day_of_week", "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "quarter"]].to_numpy()
    y = data[["label"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x_time, y, test_size=0.1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = KNeighborsClassifier(n_neighbors=7)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    with open(join(sys.path[0], path, Model_file_name), "wb") as f:
        pickle.dump(model, f)

    try:
        with open(join(sys.path[0], path, file_accuracies), "wb") as f2:
            df = pd.read_csv(f2, sep=",")
            df['KNN'] = acc
            df.to_csv(file_accuracies)
    except:
        d = [{'KNN': acc}]
        df = pd.DataFrame(data=d)
        df.to_csv(join(path, file_accuracies), index=False)


def predict_model(path, timestamp):
    centroids = pd.read_csv(join(sys.path[0], path, file_cluster_centroids)).to_numpy()
    # Load model
    model = pickle.load(open(join(sys.path[0], path, Model_file_name), 'rb'))
    result = model.predict_proba(extract_single(timestamp))
    points = sorted(
        [{'latitude': lat, 'longitude': long, 'radius': r, 'confidence': conf} for lat, long, r, conf in zip(
            centroids[:, 0], centroids[:, 1], centroids[:, 2], result.reshape(result.shape[1]))], key=lambda x: x['confidence'],
        reverse=True)
    return points

