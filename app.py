import threading
from pathlib import Path
from flask_ngrok import run_with_ngrok
import pandas as pd
from flask import Flask, jsonify, request
from os import path
import logging
import http3
import JSONtoCSV as jtc
import data_extractor as idt
from data_extractor import filename
from Models import DeepModel, NNModel, KNN

app = Flask(__name__)
run_with_ngrok(app)
OnPointAPI_URL = "https://onpoint-backend.herokuapp.com/api/"
getHist = "locationhistories/getLocationHist/"
client = http3.AsyncClient()


@app.route('/index')
def hello():
    return "OnPoint ML Engine"


async def getHistory(userId):
    keys = "?order=start&count=-1"
    req = OnPointAPI_URL+getHist+userId+keys
    res = await client.get(req)
    hist = jsonify(res.text)
    filepath = Path("/users/"+userId+"/history.csv")
    jtc.convert(hist, filepath)

def userExists(userId):
    dir = Path("/users/"+userId+"/")
    if not path.exists(dir):
        thread = threading.Thread(target=getHistory, args=(userId))
        thread.daemon = True
        thread.start()
        return False*3
    else:
        return True, path.exists(dir / filename), path.exists(dir / "accuracies.csv")




@app.route('/user/<string:userId>/predict', methods=['GET', 'POST'])
def predict(userId):
    logging.debug("predict request for user: ", userId)
    timestamp = request.args.get('timestamp')
    dir = Path("/users/"+userId+"/")
    arr = userExists(userId)
    if not arr[0]:
        idt.prepare_data(dir)
    if not arr[1]:
        KNN.train_model()
        DeepModel.train_model()
        NNModel.train_model()
    resJson = None
    df = pd.read_csv(dir/"accuracies.csv")
    if df['KNN']>df['NN'] and df['KNN']>df['DEEP']:
        resJson = KNN.predict_model(dir, timestamp)
    elif df['NN']>df['KNN'] and df['NN']>df['DEEP']:
        NNModel.predict_model(dir, timestamp)
    else:
        DeepModel.predict_model(dir, timestamp)


if __name__ == '__main__':
    app.run()