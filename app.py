import asyncio
import errno
import json
import os
import sys
import threading
import time

import requests
from flask_ngrok import run_with_ngrok
import pandas as pd
from flask import Flask, jsonify, request
from os import path
from os.path import join
import logging
import http3
import JSONtoCSV as jtc
from data_extractor import file_extracted_data, file_accuracies, prepare_data
from Models import DeepModel, NNModel, KNN

app = Flask(__name__)
run_with_ngrok(app)
ngrok_url = ""
OnPointAPI_URL = "https://onpoint-backend.herokuapp.com/api/"
getHist = "locationhistories/getLocationHist/"
client = http3.AsyncClient()


@app.route('/index')
def hello():
    return "OnPoint ML Engine"


def getHistory(userId):
    keys = "?order=start&count=-1"
    req = OnPointAPI_URL + getHist + userId + keys
    res = requests.get(req)
    hist = json.dumps(res.json())
    file_path = join(sys.path[0], "users", userId, "history.csv")
    jtc.convert(hist, file_path)


def user_exists(userId):
    file_dir = join(sys.path[0], "users")
    if not path.exists(join(file_dir, userId)):
        try:
            os.mkdir(file_dir)
            file_dir = join(file_dir, userId)
            os.mkdir(file_dir)
            getHistory(userId)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        else:
            print("Successfully created the directory %s " % file_dir)

        return False, False
    else:
        return True, path.exists(join(file_dir, file_extracted_data))


def setMLurl():
    logging.debug("sending URL")
    os.system("curl  http://localhost:4040/api/tunnels > tunnels.json")

    with open('tunnels.json') as data_file:
        data_json = json.load(data_file)

    msg = "|"
    for i in data_json['tunnels']:
        msg = msg + i['public_url'] + "|"
    http3.post(OnPointAPI_URL + "prediction/setMLurl?key=" + str(msg.split("|")[1]))
    pass


@app.route('/user/<string:userId>/predict', methods=['GET', 'POST'])
def predict(userId):
    logging.debug("predict request for user: ", userId)
    timestamp = int(request.args.get('timestamp'))
    file_dir = join(sys.path[0], "users", userId)
    arr = user_exists(userId)
    if not arr[0]:
        prepare_data(file_dir)
    if not arr[1]:
        KNN.train_model(file_dir)
        # DeepModel.train_model(file_dir)
        NNModel.train_model(file_dir)
    df = pd.read_csv(join(file_dir, file_accuracies))
    # if df['KNN'] > df['NN'] and df['KNN'] > df['DEEP']:
    #     points = KNN.predict_model(file_dir, timestamp)
    # elif df['NN'] > df['KNN'] and df['NN'] > df['DEEP']:
    #     points = NNModel.predict_model(file_dir, timestamp)
    # else:
    #     points = DeepModel.predict_model(file_dir, timestamp)
    points = KNN.predict_model(file_dir, timestamp)
    res_json = {'userId':userId, 'points':points, 'timestamp':timestamp}
    print(str(res_json))
    return res_json


if __name__ == '__main__':
    t = threading.Thread(target=app.run, args=())
    t.start()
    b = False
    while not b:
        if http3.get("http://localhost:4040").status_code == 200:
            b = True
            setMLurl()
        time.sleep(0.5)
