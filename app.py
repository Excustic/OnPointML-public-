#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, OnPoint Project"
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import errno
import json
import os
import sys
from threading import Thread
import time
import requests
from flask_ngrok import run_with_ngrok
import pandas as pd
from flask import Flask, request
from os import path
from os.path import join
import json_to_csv as jtc
from data_extractor import file_extracted_data, file_accuracies, file_location_history, prepare_data
from Models import DeepModel, NNModel, KNN

app = Flask(__name__)
run_with_ngrok(app)
ngrok_url = ""
OnPointAPI_URL = "https://onpoint-backend.herokuapp.com/api/"
getHist = "locationhistories/getLocationHist/"
getLength = "locationhistories/ArrayLength/"
update_threshold = 5000


def update_dir(user_id):
    # prepare count
    user_dir = join(sys.path[0], "users", user_id)
    df = pd.read_csv(join(user_dir, file_location_history), sep=',', header=None)
    old_len = len(df.index)
    new_len = requests.get(OnPointAPI_URL + getLength + user_id).text.split(' ')
    new_len = int(new_len[len(new_len) - 1])
    if new_len - old_len > update_threshold:
        payload = {"order": "end", "count": str(new_len - old_len)}
        # call for getHist with count
        req = OnPointAPI_URL + getHist + user_id
        headers = {'Content-Type': 'application/json'}
        try:
            hist = json.dumps(requests.get(req, params=payload, headers=headers, timeout=300, stream=True).json())
            file_path = join(user_dir, file_location_history)
            jtc.convert(hist, file_path, 'a')
        except Exception as e:
            print(e)
        prepare_data(user_dir)
        KNN.train_model(user_dir)
        DeepModel.train_model(user_dir)
        NNModel.train_model(user_dir)


def get_history(user_id):
    payload = {"order": "start", "count": '-1'}
    req = OnPointAPI_URL + getHist + user_id
    headers = {'Content-Type': 'application/json'}
    try:
        hist = json.dumps(requests.get(req, params=payload, headers=headers, timeout=300, stream=True).json())
        file_path = join(sys.path[0], "users", user_id, file_location_history)
        jtc.convert(hist, file_path, 'w')
    except Exception as e:
        print(e)


def update_users():
    file_dir = join(sys.path[0], "users")
    if not path.exists(file_dir):
        try:
            os.mkdir(file_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            else:
                print("Successfully created the directory %s " % file_dir)

    users = os.listdir(file_dir)
    for u in users:
        Thread(target=update_dir, args=(u,)).start()


def user_exists(userId):
    file_dir = join(sys.path[0], "users")
    if not path.exists(join(file_dir, userId)):
        try:
            if not path.exists(file_dir):
                os.mkdir(file_dir)
            file_dir = join(file_dir, userId)
            os.mkdir(file_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        else:
            print("Successfully created the directory %s " % file_dir)
        get_history(userId)
        return False, False
    else:
        return True, path.exists(join(file_dir, userId, file_extracted_data))


def set_ml_url():
    print("sending URL")
    os.system("curl  http://localhost:4040/api/tunnels > tunnels.json")
    time.sleep(0.1)
    with open('tunnels.json') as data_file:
        data_json = json.load(data_file)
        if len(data_json['tunnels']) < 1:
            print(data_json)
            time.sleep(2)
            return set_ml_url()

    msg = "|"
    for i in data_json['tunnels']:
        msg = msg + i['public_url'] + "|"
    print("MLurl: ", msg)
    return requests.post(OnPointAPI_URL + "prediction/setMLurl?key=" + str(msg.split("|")[1]))


@app.route('/user/<string:userId>/predict', methods=['GET', 'POST'])
def predict(user_id):
    print("predict request for user: ", user_id)
    timestamp = int(request.args.get('timestamp'))
    file_dir = join(sys.path[0], "users", user_id)
    arr = user_exists(user_id)
    if not arr[0]:
        prepare_data(file_dir)
    if not arr[1]:
        KNN.train_model(file_dir)
        DeepModel.train_model(file_dir)
        NNModel.train_model(file_dir)
    df = pd.read_csv(join(file_dir, file_accuracies))
    if (df['KNN'] > df['NN']).bool() and (df['KNN'] > df['DEEP']).bool():
        points = KNN.predict_model(file_dir, timestamp)
    elif (df['NN'] > df['KNN']).bool() & (df['NN'] > df['DEEP']).bool():
        points = NNModel.predict_model(file_dir, timestamp)
    else:
        points = DeepModel.predict_model(file_dir, timestamp)
    res_json = {'userId': user_id, 'points': points, 'timestamp': timestamp}
    print(str(res_json))
    return res_json


if __name__ == '__main__':
    t = Thread(target=app.run, args=())
    t.start()
    update_users()
    b = False
    while not b:
        if requests.get("http://localhost:4040").status_code == 200:
            b = True
            set_ml_url()
        time.sleep(0.8)
