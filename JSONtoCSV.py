import pandas as pd

def convert(json, path):
    df = pd.read_json(json, convert_dates=False)
    df.to_csv(path, index=None, date_format=None)
