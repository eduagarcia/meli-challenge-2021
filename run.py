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

DATASET_PATH = './dataset'
TEST_DATA_FILENAME = 'test_data.csv'
TRAIN_DATA_FILENAME = 'train_data.parquet'
TRAIN_DATA_PROCESSED_FILENAME = 'train_sku_feature_data.parquet'
ITEM_DATA_FILEPATH = './dataset/items_static_metadata_full.jl'

def save_results(data):
    logger_file = 'submissions.csv'
    
    if not os.path.exists('./models/hist'):
        os.makedirs('./models/hist')
    
    model_file = os.path.join('./models', f"{data['original_filename']}.py")
    out_model_file = os.path.join('./models/hist', f"{data['original_filename']}-submission-{data['unix_time']}.py")
    
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

def train_and_evaluate(model_name, save_result=True):
    date_start = datetime.now()
    all_times = []
    results = []
    
    time_data = []
    start_time = time.time()

    dataset_current_path = DATASET_PATH

    test_data_filepath = os.path.join(dataset_current_path, TEST_DATA_FILENAME)
    #ground_truth_filepath = os.path.join(dataset_current_path, GROUND_TRUTH_FILENAME)
    #train_data_filepath = os.path.join(dataset_current_path, TRAIN_DATA_FILENAME)
    #train_data_processed_filepath = os.path.join(dataset_current_path, TRAIN_DATA_PROCESSED_FILENAME)

    logger.info(f"Loading dataset {dataset_current_path}...")


    df_test = read_df(test_data_filepath)
    #ground_truth = read_numpy(ground_truth_filepath)
    #df_item = read_df(ITEM_DATA_FILEPATH)
    #df_train = read_df(train_data_filepath)
    #df_train_processed = read_df(train_data_processed_filepath)

    time_data.append(time.time() - start_time)

    logger.info(f"Initiating strategy {model_name} on dataset {dataset_current_path}")

    strategy = models[model_name](dataset_current_path)

    time_data.append(time.time() - start_time - sum(time_data))

    logger.info("Preprocessing data...")

    strategy.prepare_data()

    time_data.append(time.time() - start_time - sum(time_data))

    logger.info("Training model...")

    strategy.train()

    time_data.append(time.time() - start_time - sum(time_data))

    logger.info(f"Making predictions on {test_data_filepath}...")

    predictions = strategy.predict(df_test)

    time_data.append(time.time() - start_time - sum(time_data))

    logger.info("Generating submission file...")

    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    sub_filepath = f'predictions/{model_name}.csv'
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(sub_filepath, index=False, header=False, float_format='%.4f')

    if not os.path.exists('submissions'):
        os.mkdir('submissions')

    filename = model_name
    date = datetime.now()
    unix_time = int(time.mktime(date.timetuple()))
    gz_filepath = './submissions/'+filename+'-'+str(unix_time)+'.csv.gz'

    logger.info(f'Calculating hash of {sub_filepath}...')
    filehash = generate_file_sha256(sub_filepath)

    logger.info(f'Compressing {sub_filepath} to {gz_filepath}...')

    with open(sub_filepath, 'rb') as f_original:
        with gzip.open(gz_filepath, 'wb') as f_gz:
            f_gz.write(f_original.read())

    time_data.append(time.time() - start_time - sum(time_data))

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
    model_name = sys.argv[-1]
    train_and_evaluate(model_name)