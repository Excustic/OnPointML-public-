import datetime as dt
import operator
import time
import tracemalloc
from os.path import join

import hdbscan
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygeohash as gh
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN, KMeans, MeanShift, OPTICS, cluster_optics_dbscan

# Transform normal timestamp to special ml features
from sklearn.neighbors._ball_tree import BallTree

file_location_history = "history.csv"
file_extracted_data = "extracted_data.csv"
file_cluster_centroids = "recycle_bin/HDBSCAN_CLUSTER_CENTROIDS.csv"
file_accuracies = "accuracies.csv"


def extract(Time):
    month, day_of_week, hour, minute, hour_sin, hour_cos, month_sin, month_cos, is_weekend, quarter = ([] for i in
                                                                                                       range(10))
    for stamp in Time:
        current_date = dt.datetime.utcfromtimestamp(stamp).strftime("%Y/%m/%d %H:%M")
        yymmdd = current_date.split(" ")[0]
        hhmm = current_date.split(" ")[1]
        # year.append(int(yymmdd.split("/")[0]))
        month.append(int(yymmdd.split("/")[1]))
        day_of_week.append(
            dt.datetime.isoweekday(dt.datetime.utcfromtimestamp(stamp)) / 7)  # monday - 1 , sunday - 7
        hour.append(int(hhmm.split(":")[0]))
        minute.append(int(hhmm.split(":")[1]))
        (sin, cos) = circular_hour(hour[len(hour) - 1], minute[len(minute) - 1])
        hour_sin.append(sin)
        hour_cos.append(cos)
        (sin, cos) = circular_month(month[len(month) - 1])
        month_sin.append(sin)
        month_cos.append(cos)
        is_weekend.append(
            1 if (day_of_week[len(day_of_week) - 1] == 5 or day_of_week[len(day_of_week) - 1] == 6) else 0)
        quarter.append(np.ceil(month[len(month) - 1] / 3) / 4)
    # year_min, year_max = year[0], year[len(year) - 1]
    # year = np.array(year)
    # if (year_min == year_max):
    #     year = (year - year_min) / year_max
    # else:
    #     year = (year - year_min) / (year_max - year_min)
    return list(zip(day_of_week, hour_sin, hour_cos, month_sin, month_cos, is_weekend, quarter))


def extract_single(Time):
    current_date = dt.datetime.utcfromtimestamp(Time).strftime("%Y/%m/%d %H:%M")
    yymmdd = current_date.split(" ")[0]
    hhmm = current_date.split(" ")[1]
    year = int(yymmdd.split("/")[0])
    month = int(yymmdd.split("/")[1])
    day_of_week = dt.datetime.isoweekday(dt.datetime.utcfromtimestamp(Time)) / 7  # monday - 1 , sunday - 7
    hour = int(hhmm.split(":")[0])
    minute = int(hhmm.split(":")[1])
    (hoursin, hourcos) = circular_hour(hour, minute)
    (monthsin, monthcos) = circular_month(month)
    is_weekend = 1 if (day_of_week == 5 or day_of_week == 6) else 0
    quarter = (np.ceil(month / 3) / 4)
    arr = np.array(list((day_of_week, hoursin, hourcos, monthsin, monthcos, is_weekend, quarter)))
    arr = arr.reshape(1, arr.shape[0])
    return arr


def transform_Y(coords):
    start_time = time.time()
    tracemalloc.start()
    kms_per_radian = 6371.0088
    epsilon = 0.2 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    print('Number of clusters: {}'.format(num_clusters))
    centermost_points = clusters.map(get_centermost_point)
    lats, lons, radius = zip(*centermost_points)
    rep_points = pd.DataFrame({'lat': lats, 'lon': lons, 'radius': radius})
    # all done, print outcome
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(message.format(len(coords), len(rep_points), 100 * (1 - float(len(rep_points)) / len(coords)),
                         time.time() - start_time))
    return rep_points.to_numpy()


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    radius = 0.0
    for point in cluster:
        rad = great_circle(point, centermost_point).km
        if rad > radius:
            radius = rad
    return tuple(centermost_point) + tuple([radius])


def circular_hour(hour, minute):
    return np.sin(float(hour + minute / 60) * 15 * np.pi / 180), np.cos(
        float(hour + minute / 60) * 15 * np.pi / 180)


def circular_month(month):
    return np.sin(month * 30 * np.pi / 180), np.cos(month * 30 * np.pi / 180)


def filter_data(X, y, Time):
    full_size = len(y)
    new_X = []
    new_y = []
    arr_zone = []
    dict_lat = {}
    dict_lon = {}
    # for x in range(len(y)):
    #     if y[x][:2] in dict_zone:
    #         dict_zone[y[x][:2]] += 1
    #     else:
    #         dict_zone[y[x][:2]] = 1
    for x in range(len(y)):
        if int(y[x, 0]) in dict_lat:
            dict_lat[int(y[x, 0])] += 1
        else:
            dict_lat[int(y[x, 0])] = 1
        if int(y[x, 1]) in dict_lon:
            dict_lon[int(y[x, 1])] += 1
        else:
            dict_lon[int(y[x, 1])] = 1
    arr_lat = sorted(dict_lat.items(), key=operator.itemgetter(1), reverse=True)
    arr_lon = sorted(dict_lon.items(), key=operator.itemgetter(1), reverse=True)
    print("countries:", arr_zone)
    lat_country = []
    lon_country = []
    for lat in dict_lat:
        if dict_lat[lat] / full_size > 0.1:
            lat_country.append(lat)
    for lon in dict_lon:
        if dict_lon[lon] / full_size > 0.1:
            lon_country.append(lon)
    travel = 0
    out_of_country = 0
    for x in range(len(y)):
        if int(y[x, 0]) in lat_country and int(y[x, 1]) in lon_country:
            if x > 0 and great_circle(y[x], y[x - 1]).km / ((Time[x] - Time[x - 1]) / 3600) < 30.0 or x == 0:
                new_X.append(X[x])
                new_y.append(y[x])
            else:
                travel += 1
        else:
            out_of_country += 1
    print('eliminated ', travel, ' traveling points and', out_of_country, ' out of bounds')
    return np.array(new_X), np.array(new_y)


def Balltree_CLUSTER(coords):
    earth_radius = 6371000  # meters in earth
    test_radius = 100  # meters

    test_points = coords
    test_points_rad = [[x[0] * np.pi / 180, x[1] * np.pi / 180] for x in test_points]

    tree = BallTree(np.array(test_points_rad), metric='haversine')

    results = tree.query_radius(test_points, r=test_radius / earth_radius, return_distance=True)
    print(results)
    return results


def HDBSCAN_CLUSTER(coords, path):
    tracemalloc.start()
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=int(coords.shape[0] / 100), algorithm='prims_kdtree',
                                core_dist_n_jobs=-2, alpha=1.1)
    clusterer.fit(coords)
    labels = clusterer.labels_
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    palette = sns.color_palette(n_colors=len(set(clusterer.labels_)))
    # clusters = pd.Series([coords[cluster_labels == n] for n in range(len(clusterer.labels_))])
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(message.format(len(coords), clusterer.labels_.max() - clusterer.labels_.min() + 1,
                         100 * (1 - float(clusterer.labels_.max() - clusterer.labels_.min() + 1) / len(coords)),
                         time.time() - start_time))
    # cluster_colors = [sns.desaturate(palette[col], col/len(set(clusterer.labels_)))
    #                   if col >= 0 else (0.1, 0.7, 0.7) for col in
    #                   clusterer.labels_]
    # s = [0.1 if clusterer.labels_[i] == -1 else 25 for i in range(len(clusterer.labels_))]
    # plt.scatter(coords[:, 1], coords[:, 0], c=cluster_colors, s=s)
    # plt.show()
    num_clusters = len(set(clusterer.labels_)) - 1
    clusters = pd.Series([coords[labels == n] for n in range(0, num_clusters)])
    centers = clusters.map(get_centermost_point).array
    centers = np.array([[a[0], a[1], a[2]] for a in centers])
    np.savetxt(join(path, file_cluster_centroids), centers, fmt="%10.8f", delimiter=',')
    return labels


def DBSCAN_CLUSTER(coords):
    start_time = time.time()
    tracemalloc.start()
    kms_per_radian = 6371.0088
    epsilon = 0.05 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=300, algorithm='ball_tree', metric='haversine', n_jobs=-1).fit(
        np.radians(coords))
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters - 1)])
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(message.format(len(coords), num_clusters, 100 * (1 - float(num_clusters) / len(coords)),
                         time.time() - start_time))
    centers = clusters.map(get_centermost_point)
    print(centers)
    palette = sns.color_palette(n_colors=num_clusters)
    cluster_colors = [sns.desaturate(palette[col], col / num_clusters)
                      if col >= 0 else (0.1, 0.7, 0.7) for col in
                      cluster_labels]
    s = [0.1 if cluster_labels[i] == -1 else 25 for i in range(len(cluster_labels))]
    plt.scatter(coords[:, 1], coords[:, 0], c=cluster_colors, s=s)
    plt.show()
    # return num_clusters


def OPTICS_CLUSTER(coords):
    tracemalloc.start()
    start_time = time.time()
    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian
    max_epsilon = 1.5 / kms_per_radian
    # cluster_method='dbscan', eps=epsilon, max_eps=max_epsilon, metric='haversine', min_samples=2,
    clustering = OPTICS(n_jobs=-1, min_samples=50).fit(coords)
    labels = cluster_optics_dbscan(reachability=clustering.reachability_,
                                   core_distances=clustering.core_distances_,
                                   ordering=clustering.ordering_, eps=0.5)
    # labels = clustering.labels_
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(message.format(len(coords), labels.max() - labels.min() + 1,
                         100 * (1 - float(labels.max() - labels.min() + 1) / len(coords)),
                         time.time() - start_time))
    num_clusters = len(set(labels))
    clusters = pd.Series([coords[labels == n] for n in range(num_clusters - 1)])
    centers = clusters.map(get_centermost_point).array
    centers = np.array([[a[0], a[1]] for a in centers])
    np.savetxt("OPTICS_CLUSTER_CENTROIDS.csv", centers, fmt="%10.8f", delimiter=',')
    print(centers)


def GEOHASH_CLUSTER(coords, threshold, precision):
    if 1 < threshold < 0:
        raise ValueError("threshold supposed to be float between 0 and 1")
    start_time = time.time()
    tracemalloc.start()
    geohash = []
    dict = {}
    hash_clusters = []
    for i in range(len(data['Lat'])):
        geohash_obj = gh.encode(float(data['Lat'][i]), float(data['Long'][i]), precision=precision)
        geohash.append(geohash_obj)
        if geohash_obj not in dict:
            dict[geohash_obj] = 1
        else:
            dict[geohash_obj] += 1
    arr = Counter(dict).most_common()
    for item in arr:
        if (item[1] / len(coords) > threshold):
            hash_clusters.append(item[0])
    if len(hash_clusters) == 0:
        hash_clusters.append(arr[0][0])
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(message.format(len(coords), len(hash_clusters), 100 * (1 - float(len(hash_clusters)) / len(coords)),
                         time.time() - start_time))
    plt.scatter(coords[:, 1], coords[:, 0], c='black', s=2)
    gh.decode(hash_clusters[0])
    plt.scatter([gh.decode(d)[1] for d in hash_clusters], [gh.decode(d)[0] for d in hash_clusters], marker='s', c='red',
                s=20, alpha=0.8)
    plt.show()
    # return hash_clusters


def CLUSTER_MEANSHIFT(coords):
    start_time = time.time()
    tracemalloc.start()
    ms = MeanShift()
    ms.fit(coords)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    cluster_centers = ms.cluster_centers_
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(
        message.format(len(coords), len(ms.cluster_centers_), 100 * (1 - float(len(ms.cluster_centers_)) / len(coords)),
                       time.time() - start_time))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(coords[:, 1], coords[:, 0], marker='o')
    ax.scatter(cluster_centers[:, 1], cluster_centers[:, 0],
               marker='x', color='red',
               s=10, linewidth=5, alpha=0.5)
    plt.show()
    return len(cluster_centers)


def kmeans_cluster(coords, k):
    # Train model
    start_time = time.time()
    tracemalloc.start()
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(coords)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
    print(message.format(len(coords), k, 100 * (1 - float(k) / len(coords)),
                         time.time() - start_time))
    # Plot clusters
    plt.figure()
    centers = kmeans.cluster_centers_
    plt.scatter(
        centers[:, 1],
        centers[:, 0],
        c='red',
        s=20,
        alpha=0.5
    )
    plt.scatter(coords[:, 1], coords[:, 0], s=2, c='black')
    plt.show()


def prepare_data(path):
    data = pd.read_csv(join(path, file_location_history), sep=",")

    Time = data['timestamp']
    coords = np.column_stack([data["latitude"], data["longitude"]])

    X_Time = extract(Time)
    X_Time = np.array(X_Time)

    y = np.array(coords)

    filtered_x, filtered_y = filter_data(X_Time, y, Time)
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)

    label = HDBSCAN_CLUSTER(filtered_y, path)

    filtered_x = filtered_x[label != -1]
    filtered_y = filtered_y[label != -1]
    label = label[label != -1]

    np.set_printoptions(suppress=True)
    transformed_data = np.column_stack([filtered_x, filtered_y, label])
    header_str = "day_of_week,hour_sin,hour_cos,month_sin,month_cos,is_weekend,quarter,lat,long,label"
    format_str = []

    for i in range(transformed_data.shape[1] - 1):
        format_str.append("%10.8f")

    format_str.append("%d")
    header_str_arr = header_str.split(',')
    dtype = []

    for i in range(len(header_str_arr) - 1):
        dtype.append((header_str_arr[i], float))
    dtype.append((header_str_arr[i + 1], int))
    ab = np.zeros(filtered_y.shape[0], dtype=dtype)

    for i in range(len(header_str_arr)):
        ab[header_str_arr[i]] = transformed_data[:, i]

    np.savetxt(join(path, file_extracted_data), ab, delimiter=',', fmt=format_str, header=header_str, comments='')
