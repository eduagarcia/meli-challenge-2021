#
# Fix: remove sum of sold_quantity == 0 from training data
#

import numpy as np
import json
import os
import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=50)

from xgboost import XGBClassifier
from tsfresh.utilities.dataframe_functions import impute

from model import Model
from utils import read_df

x_df = None
x_test_df = None

def impute_pool_train(column):
    return impute(x_df[column].to_frame())

def impute_pool_test(column):
    return impute(x_test_df[column].to_frame())

class XGBoostFeaturesV3_1(Model):
    model_name = 'xgboost_features_v3_1_tsfresh_all'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        global x_df
        
        df_train_x_features_tsfresh = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_x_processed_tsfresh']))
        df_train_y_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_y_processed']))
        
        x_df = df_train_x_features_tsfresh.sample(frac=1, random_state=42).copy()
        y_df = df_train_y_features.set_index('sku').loc[x_df.index].copy()
        
        y_sold_quantity_series = np.array(list(y_df['sold_quantity_series'].apply(json.loads).values))
        y_cumsum = y_sold_quantity_series.cumsum(axis=1)

        SELECT_N = 1

        index_list = []
        y_ts = []
        y_td = []

        for i, t in zip(y_df.index, y_cumsum):
            target_stocks, target_dates = np.unique(t, return_index=True)
            if target_stocks[0] == 0:
                target_stocks = target_stocks[1:]
                target_dates = target_dates[1:]
            size = len(target_stocks)
            index = np.arange(size)
            if size > SELECT_N:
                np.random.shuffle(index)
                index = index[:SELECT_N]

            for target_stock, target_date in zip(target_stocks[index], target_dates[index]):
                index_list.append(i)
                y_ts.append(target_stock)
                y_td.append(target_date)

        index_list = np.array(index_list)
        y_ts = np.array(y_ts)
        y_td = np.array(y_td)
        
        x_df_imputed = []
        with Pool(96) as p:
            for data in tqdm(p.imap(impute_pool_train, x_df.columns), total=len(x_df.columns)):
                x_df_imputed.append(data)

        x_df = pd.concat(x_df_imputed, axis=1) 
        
        x_df_filtered = x_df.loc[index_list]
        
        X = np.concatenate((x_df_filtered.values, np.reshape(y_ts, (-1, 1))), axis=1)
        y = y_td
        
        self.prepared_dataset = (X, y)
        
    def train(self):
        X, y = self.prepared_dataset
        self.model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=0, tree_method='gpu_hist',
                    objective='multi:softprob', num_class=30, use_label_encoder=False,
                    gpu_id=3)
        self.model.fit(X, y)
    
    def predict(self, df_test):
        global x_test_df
        df_test = read_df(df_test)
        df_test_fromtrain_x_features_tsfresh = read_df(os.path.join(self.dataset_path, self.default_paths['test_fromtrain_data_x_processed_tsfresh']))
        
        x_test_df = df_test_fromtrain_x_features_tsfresh.loc[df_test['sku']].copy()
        
        X_test = np.concatenate((x_test_df.values, np.reshape(df_test['target_stock'].values, (-1, 1))), axis=1)
        
        x_test_df_imputed = []
        with Pool(96) as p:
            for data in tqdm(p.imap(impute_pool_test, x_test_df.columns), total=len(x_test_df.columns)):
                x_test_df_imputed.append(data)

        x_test_df = pd.concat(x_test_df_imputed, axis=1)
        
        X_test = np.concatenate((x_test_df.values, np.reshape(df_test['target_stock'].values, (-1, 1))), axis=1)
        
        preds = self.model.predict_proba(X_test)
        probabilities = (preds/preds.sum(axis=1)[:,None]).round(4)
        
        return probabilities

