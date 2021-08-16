import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import hashlib
import gzip

from models import models
from utils import read_df, read_numpy
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def save_results(data):
    logger_file = 'submissions.csv'

    if not os.path.exists(logger_file):
        df = pd.DataFrame([data])
        df.to_csv(logger_file, index=False)
    else:
        df = pd.read_csv(logger_file)
        df = df.append(data, ignore_index=True)
        df.to_csv(logger_file, index=False)
        
def generate_file_sha256(filepath, blocksize=2**20):
    m = hashlib.sha256()
    with open(filepath , "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()

def ensemble(model_names, save_result=True):
    data = []
    
    for model_name in model_names:
        data.append(read_numpy(os.path.join('./predictions', f'{model_name}.csv')))
        
    result = np.zeros(data[0].shape)
    for submission in data:
        result += submission
    result = result/len(model_names)

    predictions = (result / result.sum(axis=1)[:,None]).round(4)
    
    filename = "_".join(model_names) + "_mean"
    
    if not os.path.exists('predictions/ensemble'):
        os.makedirs('predictions/ensemble')

    sub_filepath = f'predictions/ensemble/{filename}.csv'
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(sub_filepath, index=False, header=False)

    if not os.path.exists('submissions/ensemble'):
        os.makedirs('submissions/ensemble')

    date = datetime.now()
    unix_time = int(time.mktime(date.timetuple()))
    gz_filepath = 'submissions/ensemble/'+filename+'-'+str(unix_time)+'.csv.gz'

    logger.info(f'Calculating hash of {sub_filepath}...')
    filehash = generate_file_sha256(sub_filepath)

    logger.info(f'Compressing {sub_filepath} to {gz_filepath}...')

    with open(sub_filepath, 'rb') as f_original:
        with gzip.open(gz_filepath, 'wb') as f_gz:
            f_gz.write(f_original.read())

    data= {
        'sha256': filehash,
        'datetime': date,
        'unix_time': unix_time,
        'original_filepath': sub_filepath,
        'original_filename': filename,
        'saved_filepath': gz_filepath,
        'result': 0.0
    }

    if save_result:
        save_results(data)
    return data

if __name__ == "__main__":
    model_names = sys.argv[1:]
    ensemble(model_names)