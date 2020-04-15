#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, OnPoint Project"
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import pandas as pd

def convert(json, path, mode):
    df = pd.read_json(json, convert_dates=False, orient='list')
    df = df.drop_duplicates(subset='timestamp')
    df.to_csv(path, index=None, date_format=None, mode=mode)

