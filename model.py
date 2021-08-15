from datetime import datetime
from utils import read_df
import os

from evaluate import evaluate_rps

class Model():
    default_paths = {
        'train_data': 'train_data.parquet',
        'item_data': './dataset/items_static_metadata_full.jl',
        'train_data_processed': 'train_data_features.parquet',
        'train_data_x': 'train_data_x.parquet',
        'train_data_x_processed': 'train_data_x_features.parquet',
        'train_data_y': 'train_data_y.parquet',
        'train_data_y_processed': 'train_data_y_features.parquet',
        'test_fromtrain_data_x': 'test_fromtrain_data_last29.parquet',
        'test_fromtrain_data_x_processed': 'test_fromtrain_data_last29_features.parquet',
        'train_data_processed_tsfresh': 'train_data_features_tsfresh.parquet',
        'train_data_x_processed_tsfresh': 'train_data_x_features_tsfresh.parquet',
        'test_fromtrain_data_x_processed_tsfresh': 'test_fromtrain_data_last29_features_tsfresh.parquet'
    }
    
    def __init__(self, name, dataset_path):
        self.name = name
        self.dataset_path = dataset_path
        self.date = datetime.now()
        
    def prepare_data(self):
        self.df_train = read_df(os.path.join(self.dataset_path, self.default_paths['train_data']))
        self.df_item = read_df(self.default_paths['item_data'])
        self.df_train_processed = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_processed']))
        
        self.prepared_dataset = self.df_train_processed
        
    def train(self):
        pass
    
    def predict(self, df_test):
        return df_test
    
    def evaluate(self, prediction, ground_truth):
        return evaluate_rps(prediction, ground_truth)