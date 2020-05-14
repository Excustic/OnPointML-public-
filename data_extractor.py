#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, OnPoint Project"
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import datetime as dt
import time
import tracemalloc
from os.path import join
from random import randint
from threading import Thread
import folium
import hdbscan
import pytz
import numpy as np
import pandas as pd
from tzwhere import tzwhere
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

# default naming for different csv files
file_location_history = "history.csv"
file_extracted_data = "extracted_data.csv"
file_cluster_centroids = "HDBSCAN_CLUSTER_CENTROIDS.csv"
file_accuracies = "accuracies.csv"
file_home_cluster = "home_cluster.csv"


# extract data features from a timestamp vector
def extract(data_time, timezone):
    month, day_of_week, hour, minute, hour_sin, hour_cos, month_sin, month_cos, is_weekend, quarter = ([] for _ in
                                                                                                       range(10))
    for stamp in data_time:
        # convert timestamp to a datetime object
        current_date = dt.datetime.fromtimestamp(stamp, timezone)

        month.append(current_date.month)
        day_of_week.append(
            dt.datetime.isoweekday(current_date) / 7)  # monday - 1 , sunday - 7
        hour.append(current_date.hour)
        minute.append(current_date.minute)
        (sin, cos) = circular_hour(hour[len(hour) - 1], minute[len(minute) - 1])
        hour_sin.append(sin)
        hour_cos.append(cos)
        (sin, cos) = circular_month(month[len(month) - 1])
        month_sin.append(sin)
        month_cos.append(cos)
        is_weekend.append(
            1 if (day_of_week[len(day_of_week) - 1] == 5 / 7 or day_of_week[len(day_of_week) - 1] == 6 / 7) else 0)
        quarter.append(np.ceil(month[len(month) - 1] / 3) / 4)

    return list(zip(day_of_week, hour_sin, hour_cos, month_sin, month_cos, is_weekend, quarter))


# extract data features from a single timestamp
def extract_single(data_time, timezone):
    # convert timestamp to a datetime object
    current_date = dt.datetime.fromtimestamp(data_time, timezone)

    month = current_date.month
    day_of_week = dt.datetime.isoweekday(dt.datetime.utcfromtimestamp(data_time)) / 7  # monday - 1 , sunday - 7
    hour = current_date.hour
    minute = current_date.minute
    (hoursin, hourcos) = circular_hour(hour, minute)
    (monthsin, monthcos) = circular_month(month)
    is_weekend = 1 if (day_of_week == 5 / 7 or day_of_week == 6 / 7) else 0
    quarter = (np.ceil(month / 3) / 4)

    # wrap it in a two dimensional numpy array
    arr = np.array(list((day_of_week, hoursin, hourcos, monthsin, monthcos, is_weekend, quarter)))
    arr = arr.reshape(1, arr.shape[0])

    return arr


# returns a real point from the cluster that is closest to the true centroid
def get_centermost_point(cluster):
    # get true centroid
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    # compare each point to the centroid and retrieve the closest one
    centermost_point = min(cluster, key=lambda mpoint: great_circle(mpoint, centroid).m)
    radius = 0.0
    # find the distance of the farthest point from the centermost_point
    for point in cluster:
        rad = great_circle(point, centermost_point).km
        if rad > radius:
            radius = rad
    return tuple(centermost_point) + tuple([radius])


# turn hour to a harmonic feature
def circular_hour(hour, minute):
    return np.sin(float(hour + minute / 60) * 15 * np.pi / 180), np.cos(
        float(hour + minute / 60) * 15 * np.pi / 180)


# turn month to a harmonic feature
def circular_month(month):
    return np.sin(month * 30 * np.pi / 180), np.cos(month * 30 * np.pi / 180)


# filter unwanted data
def filter_data(data_time, y):
    threshold = 0.1  # threshold to cut off data
    maximum_speed = 30.0
    full_size = len(y)
    new_time = []
    new_y = []
    dict_lat = {}
    dict_lon = {}
    # count coordinates by units digit (state-size ~ 111km)
    for x in range(len(y)):
        if int(y[x, 0]) in dict_lat:
            dict_lat[int(y[x, 0])] += 1
        else:
            dict_lat[int(y[x, 0])] = 1
        if int(y[x, 1]) in dict_lon:
            dict_lon[int(y[x, 1])] += 1
        else:
            dict_lon[int(y[x, 1])] = 1
    lat_country = []
    lon_country = []
    # get rid of zones that don't hold enough coordinates
    for lat in dict_lat:
        if dict_lat[lat] / full_size > threshold:
            lat_country.append(lat)
    for lon in dict_lon:
        if dict_lon[lon] / full_size > threshold:
            lon_country.append(lon)
    # add a range of +- 1 for situations where the users crosses zones frequently
    arr = []
    for lat in lat_country:
        arr.append(lat)
        arr.append(lat + 1)
        arr.append(lat - 1)
    lat_country = arr
    arr = []
    for lon in lon_country:
        arr.append(lon)
        arr.append(lon + 1)
        arr.append(lon - 1)
    lon_country = arr
    travel = 0
    out_of_country = 0
    # filter points that are outside of home country or when user is using a vehicle
    for x in range(len(y)):
        if int(y[x, 0]) in lat_country and int(y[x, 1]) in lon_country:
            if x > 0 and great_circle(y[x], y[x - 1]).km / ((data_time[x] - data_time[x - 1]) / 3600) < maximum_speed \
                    or x == 0:
                new_time.append(data_time[x])
                new_y.append(y[x])
            else:
                travel += 1
        else:
            out_of_country += 1

    print('eliminated ', travel, ' traveling points and', out_of_country, ' out of bounds')
    return np.array(new_time), np.array(new_y)


# Hierarchical Density-Based Spatial Clustering of Applications with Noise
def hdbscan_cluster(coords, path):
    # used for benchmarking performance
    tracemalloc.start()
    start_time = time.time()
    message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'

    # Choosing parameters that fit task
    # alpha makes the clustering more conservative
    clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=int(coords.shape[0] / 100), algorithm='prims_kdtree',
                                core_dist_n_jobs=-2, alpha=1.1)
    clusterer.fit(coords)
    labels = clusterer.labels_  # returns a list of integers from min (-1 is noise data) to max (number of clusters -1)
    # benchmarking results
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    print(message.format(len(coords), clusterer.labels_.max() - clusterer.labels_.min() + 1,
                         100 * (1 - float(clusterer.labels_.max() - clusterer.labels_.min() + 1) / len(coords)),
                         time.time() - start_time))

    # save clusters by choosing central point with a radius
    num_clusters = len(set(clusterer.labels_)) - 1
    clusters = pd.Series([coords[labels == n] for n in range(0, num_clusters)])
    centers = clusters.map(get_centermost_point).array
    centers = np.array([[a[0], a[1], a[2]] for a in centers])
    cluster_home = centers[max([a] for a in clusters.iteritems())[0][0]]
    colors = []
    for i in range(40):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    folium_map = folium.Map(location=[cluster_home[0], cluster_home[1]])
    c = -1
    for cluster in clusters:
        c += 1
        for point in cluster:
            folium.CircleMarker(location=point, color=colors[c]).add_to(folium_map)
    Thread(target=save_map, args=(folium_map, join(path, 'map.html'),)).start()
    np.savetxt(join(path, file_cluster_centroids), centers, fmt="%10.8f", delimiter=',')
    return labels, cluster_home


def save_map(folium_map, path):
    folium_map.save(path)


# creates extracted_data.csv, main function
def prepare_data(path):
    # retrieve data and organize to i/o arrays
    data = pd.read_csv(join(path, file_location_history), sep=",")

    data_time = data['timestamp']
    coords = np.column_stack([data["latitude"], data["longitude"]])

    y = np.array(coords)

    # filter irrelevant points
    filtered_x, filtered_y = filter_data(data_time, y)
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)

    # perform clustering
    label, home = hdbscan_cluster(filtered_y, path)

    # extract time features
    tz = tzwhere.tzwhere()
    timezone_str = pytz.timezone(tz.tzNameAt(home[0], home[1]))  # get timezone of home cluster
    x_time = extract(filtered_x, timezone_str)
    x_time = np.array(x_time)

    # filter out noise data that clustering algorithm can't assign to a cluster
    filtered_x = x_time[label != -1]
    filtered_y = filtered_y[label != -1]
    label = label[label != -1]

    # save user's home location in a different file
    np.set_printoptions(suppress=True)
    home = np.array(home)
    np.savetxt(join(path, file_home_cluster), home, delimiter=',', comments='')

    # save the extracted data to a csv file
    transformed_data = np.column_stack([filtered_x, filtered_y, label])
    header_str = "day_of_week,hour_sin,hour_cos,month_sin,month_cos,is_weekend,quarter,lat,long,label"
    format_str = []

    for i in range(transformed_data.shape[1] - 1):
        format_str.append("%10.8f")

    format_str.append("%d")
    header_str_arr = header_str.split(',')
    dtypes = []

    for i in range(len(header_str_arr) - 1):
        dtypes.append((header_str_arr[i], float))
    dtypes.append((header_str_arr[i + 1], int))
    final = np.zeros(filtered_y.shape[0], dtype=dtypes)

    for i in range(len(header_str_arr)):
        final[header_str_arr[i]] = transformed_data[:, i]

    np.savetxt(join(path, file_extracted_data), final, delimiter=',', fmt=format_str, header=header_str, comments='')
