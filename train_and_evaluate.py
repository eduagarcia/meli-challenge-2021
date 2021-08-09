import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd

from models import models
from utils import read_df, read_numpy
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DATASET_PATH = './dataset/processed/train_v1'
TEST_DATA_FILENAME = 'test_data.csv'
GROUND_TRUTH_FILENAME = 'test_ground_truth.npy'
TRAIN_DATA_FILENAME = 'train_data.parquet'
TRAIN_DATA_PROCESSED_FILENAME = 'train_sku_feature_data.parquet'

def save_results(data):
    logger_file = 'predictions.csv'
    
    if not os.path.exists(logger_file):
        df = pd.DataFrame([data])
        df.to_csv(logger_file, index=False)
    else:
        df = pd.read_csv(logger_file)
        df = df.append(data, ignore_index=True)
        df.to_csv(logger_file, index=False)

def train_and_evaluate(model_name, dataset_indexes, save_result=True):
    date_start = datetime.now()
    all_times = []
    results = []
    
    for dataset_index in dataset_indexes:
        time_data = []
        start_time = time.time()

        dataset_current_path = os.path.join(DATASET_PATH, str(dataset_index))

        test_data_filepath = os.path.join(dataset_current_path, TEST_DATA_FILENAME)
        ground_truth_filepath = os.path.join(dataset_current_path, GROUND_TRUTH_FILENAME)
        train_data_filepath = os.path.join(dataset_current_path, TRAIN_DATA_FILENAME)
        train_data_processed_filepath = os.path.join(dataset_current_path, TRAIN_DATA_PROCESSED_FILENAME)

        logger.info(f"Loading dataset {dataset_current_path}...")

        df_test = read_df(test_data_filepath)
        ground_truth = read_numpy(ground_truth_filepath)
        df_train = read_df(train_data_filepath)
        df_train_processed = read_df(train_data_processed_filepath)

        time_data.append(time.time() - start_time)

        logger.info(f"Initiating strategy {model_name} on dataset {dataset_current_path}")

        strategy = models[model_name](df_train, df_train_processed)

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

        logger.info("Evaluating...")

        result = strategy.evaluate(predictions, ground_truth)

        time_data.append(time.time() - start_time - sum(time_data))

        logger.info(f"Result of model {model_name} on index {str(dataset_index)}: {str(result)}")
        
        logger.info(f"Took {sum(time_data)} seconds. Time data of model {model_name} on index {str(dataset_index)}: {str(json.dumps(time_data))}")
        
        results.append(result)
        all_times.append(time_data)
    
    all_times_mean = np.zeros(len(all_times[0]))
    for time_data in all_times:
        all_times_mean += np.array(time_data)
    all_times_mean = all_times_mean/len(all_times)
    
    data = {
        'date_start': date_start,
        'date_end': datetime.now(),
        'model_name': model_name,
        'dataset': DATASET_PATH,
        'dataset_indexes': json.dumps(dataset_indexes),
        'result_mean': sum(results)/len(results),
        'result_per_dataset': json.dumps(results),
        'mean_time_per_dataset': all_times_mean.sum(),
        'mean_time_per_stage': json.dumps(all_times_mean.tolist())
    }
    
    if save_result:
        save_results(data)
    
    return data

if __name__ == "__main__":
    #dateset_indexes = [0, 1, 2, 3]
    dateset_indexes = [0]
    model_name = 'voted_shifted_padded_gaussian_probs'
    train_and_evaluate(model_name, dateset_indexes)