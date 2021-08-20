#
# Featuresv2 dataset
#

import numpy as np
import json
import os
import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool

from xgboost import XGBClassifier
from tsfresh.utilities.dataframe_functions import impute

from model import Model
from utils import read_df

id_column =  'sku'
date_column = 'date'

item_string_columns = ['item_title']
item_categorical_columns = ['item_domain_id', 'item_id', 'site_id', 'product_id', 'product_family_id']
item_columns = item_string_columns + item_categorical_columns

sku_numeric_columns = ['sold_quantity', 'current_price', 'minutes_active']
sku_categorical_columns = ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']
sku_columns = sku_numeric_columns + sku_categorical_columns

string_columns = item_string_columns
categorical_columns = sku_categorical_columns + item_categorical_columns
numeric_columns = sku_numeric_columns

def target_stock_features(df):
    df['target_stock__target_day_by_mean'] = df['target_stock']/df['sold_quantity__mean']
    df['target_stock__target_day_by_mean__std'] = df['sold_quantity__std']*df['target_stock__target_day_by_mean']/df['sold_quantity__mean']
    df['target_stock__last_5__target_day_by_mean'] = df['target_stock']/df['sold_quantity__last_5__mean']
    df['target_stock__last_5__target_day_by_mean__std'] = df['sold_quantity__last_5__std']*df['target_stock__last_5__target_day_by_mean']/df['sold_quantity__last_5__mean']
    
    df['target_stock__by_item_domain_id__target_day_by_mean'] = df['target_stock']/df['sold_quantity__by_item_domain_id__mean']
    df['target_stock__by_item_domain_id__target_day_by_mean__std'] = df['sold_quantity__by_item_domain_id__std']*df['target_stock__by_item_domain_id__target_day_by_mean']/df['sold_quantity__by_item_domain_id__mean']
    
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for feature in df.columns:
        if feature.startswith('target_stock__'):
            df[feature] = impute(df[[feature]])[feature]

class XGBoostFeaturesV4_1(Model):
    model_name = 'xgboost_features2_v4_1_target_features'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):        
        df_train_x_featuresv2 = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_x_processedv2']))
        df_train_y_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_y_processed']))
        
        numeric_features = []
        categorical_features = []
        positional_features = []
        counting_features = []
        date_features = []
        string_features = []
        series_features = []
        id_features = []
        for feature in df_train_x_featuresv2.columns:
            feature_components = feature.split('__')
            if ('with' in feature_components[-1]) or ('location' in feature_components[-1]):
                positional_features.append(feature)
            elif ('count' in feature_components[-1]):
                counting_features.append(feature)
            elif feature_components[-1] == 'series':
                series_features.append(feature)
            elif feature_components[0] in categorical_columns:
                categorical_features.append(feature)
            elif feature_components[0] == 'date':
                if feature == 'date__first' or feature == 'date__last':
                    date_features.append(feature)
                else:
                    positional_features.append(feature)
            elif feature_components[0] in string_columns:
                string_features.append(feature)
            elif feature_components[0] == id_column:
                id_features.append(feature)
            else:
                numeric_features.append(feature)
        
        features = categorical_features + positional_features + counting_features + numeric_features + ['target_stock']
        x_df = df_train_x_featuresv2.set_index('sku').sample(frac=1, random_state=42).copy()
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
        
        y = y_td
        
        x_df_train = x_df.loc[index_list].copy()
        x_df_train['target_stock'] = y_ts
        x_df_train = x_df_train[features]
        for feature in categorical_features:
            if feature not in features:
                continue
            x_df_train[feature] = x_df_train[feature].cat.codes
        target_stock_features(x_df_train)
        X = x_df_train.values
        
        self.prepared_dataset = (X, y)
        self.features = features
        self.categorical_features = categorical_features
    
    def train(self):
        X, y = self.prepared_dataset
        self.model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.02,
                    random_state=0, tree_method='gpu_hist',
                    objective='multi:softprob', num_class=30, use_label_encoder=False,
                    gpu_id=3)
        self.model.fit(X, y)
    
    def predict(self, df_test):
        df_test = read_df(df_test)
        df_train_featuresv2 = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_processedv2']))
        
        x_test_df_all = df_train_featuresv2.set_index('sku').loc[df_test['sku']].copy()
        x_test_df_all['target_stock'] = df_test.set_index('sku')['target_stock']
        x_test_df_all = x_test_df_all[self.features]
        for feature in self.categorical_features:
            if feature not in self.features:
                continue
            x_test_df_all[feature] = x_test_df_all[feature].cat.codes
        target_stock_features(x_test_df_all)
        X_test = x_test_df_all.values
        
        preds = self.model.predict_proba(X_test)
        probabilities = (preds/preds.sum(axis=1)[:,None]).round(4)
        
        return probabilities

