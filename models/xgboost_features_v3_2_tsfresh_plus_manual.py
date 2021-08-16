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

all_columns = ['item_domain_id', 'item_id', 'item_title', 'site_id', 'sku',
       'product_id', 'product_family_id', 'count', 'date_first', 'date_last',
       'date_diff', 'date_first_day', 'date_first_month', 'date_last_day',
       'date_last_month', 'sold_quantity_first', 'sold_quantity_last',
       'sold_quantity_sum', 'sold_quantity_mean', 'sold_quantity_std',
       'sold_quantity_min', 'sold_quantity_max', 'sold_quantity_mode',
       'sold_quantity_mode_tx', 'current_price_first', 'current_price_last',
       'current_price_sum', 'current_price_mean', 'current_price_std',
       'current_price_min', 'current_price_max', 'current_price_mode',
       'current_price_mode_tx', 'minutes_active_first', 'minutes_active_last',
       'minutes_active_sum', 'minutes_active_mean', 'minutes_active_std',
       'minutes_active_min', 'minutes_active_max', 'minutes_active_mode',
       'minutes_active_mode_tx', 'currency_first', 'currency_last',
       'currency_mode', 'currency_mode_tx', 'listing_type_first',
       'listing_type_last', 'listing_type_mode', 'listing_type_mode_tx',
       'shipping_logistic_type_first', 'shipping_logistic_type_last',
       'shipping_logistic_type_mode', 'shipping_logistic_type_mode_tx',
       'shipping_payment_first', 'shipping_payment_last',
       'shipping_payment_mode', 'shipping_payment_mode_tx',
       'minutes_active_series', 'current_price_series',
       'sold_quantity_series']

categorical_columns = ['site_id', 'item_id']
categorical_columns_with_nan = ['item_domain_id', 'product_id', 'product_family_id']
categorical_columns_composed = ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']
categorical_composed_suffix = ['_first', '_last', '_mode']

categorical_columns = categorical_columns + categorical_columns_with_nan
for category_name in categorical_columns_composed:
    for suffix in categorical_composed_suffix:
        categorical_columns.append(category_name+suffix)

numeric_columns = ['count', 'date_first_day', 'date_first_month', 'date_last_day', 'date_last_month']  
numeric_columns_composed = ['sold_quantity', 'current_price', 'minutes_active']
numeric_composed_suffix = ['_first', '_last', '_mode', '_mode_tx', '_sum', '_mean', '_std', '_min', '_max']

for numeric_name in numeric_columns_composed:
    for suffix in numeric_composed_suffix:
        numeric_columns.append(numeric_name+suffix)

categorical_to_numeric_composed_suffix = ['_mode_tx']
        
for category_name in categorical_columns_composed:
    for suffix in categorical_to_numeric_composed_suffix:
        numeric_columns.append(category_name+suffix)
        
series_columns = ['sold_quantity_series']
series_columns_headers = {}

series_columns_max = 29
for series_name in series_columns:
    series_columns_headers[series_name] = [series_name+'_'+str(i) for i in range(series_columns_max)]
    numeric_columns = numeric_columns + series_columns_headers[series_name]

manual_features = numeric_columns + categorical_columns

x_df = None
x_test_df = None

def impute_pool_train(column):
    return impute(x_df[column].to_frame())

def impute_pool_test(column):
    return impute(x_test_df[column].to_frame())

class XGBoostFeaturesV3_2(Model):
    model_name = 'xgboost_features_v3_2_tsfresh_plus_manual.py'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
    
    def _prepare_train_manual_features(self, x_df):
        x_df = x_df.copy()
        for column in numeric_columns_composed:
            x_df[column+'_std'] = x_df[column+'_std'].fillna(0)
        for column in categorical_columns_with_nan:    
            x_df[column] = x_df[column].astype(str).astype("category")
            
        categorical_pandas_order = {}

        for column in categorical_columns:    
            x_df[column] = x_df[column].astype("category")
            categorical_pandas_order[column] = x_df[column].cat.categories
        
        for column in categorical_columns:
            x_df[column] = x_df[column].cat.codes
            
        for column in series_columns:
            def series_to_df(data):
                result = np.ones(series_columns_max)*-1
                data = json.loads(data)
                index_limit = len(data) if len(data) <= series_columns_max else series_columns_max
                result[-index_limit:] = data[-index_limit:]
                #return result
                return pd.Series(result, index=series_columns_headers[column])

            series_x_df = x_df[column].parallel_apply(series_to_df)

            x_df = x_df.join(series_x_df)
            
        self.categorical_pandas_order = categorical_pandas_order
        return x_df[manual_features]
    
    def _prepare_test_manual_features(self, test_x_df):
        test_x_df = test_x_df.copy()
        for column in numeric_columns_composed:
            test_x_df[column+'_std'] = test_x_df[column+'_std'].fillna(0)
        for column in categorical_columns_with_nan:    
            test_x_df[column] = test_x_df[column].astype(str).astype("category")
        for column in categorical_columns:
            test_x_df[column] = test_x_df[column].astype("category")
            categorical_order = list(self.categorical_pandas_order[column])
            if len(test_x_df[column].cat.categories) != len(categorical_order):
                for cat in set(test_x_df[column].cat.categories) - set(categorical_order):
                        categorical_order.append(cat)
            test_x_df[column] = test_x_df[column].cat.set_categories(categorical_order, ordered=True)
            test_x_df[column] = test_x_df[column].cat.codes
        for column in series_columns:
            def series_to_df(data):
                result = np.ones(series_columns_max)*-1
                data = json.loads(data)
                index_limit = len(data) if len(data) <= series_columns_max else series_columns_max
                result[-index_limit:] = data[-index_limit:]
                #return result
                return pd.Series(result, index=series_columns_headers[column])

            series_x_test_df = test_x_df[column].parallel_apply(series_to_df)
            test_x_df = test_x_df.join(series_x_test_df)
        return test_x_df[manual_features]
        
    def prepare_data(self):
        global x_df
        
        df_train_x_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_x_processed']))
        df_train_x_features_tsfresh = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_x_processed_tsfresh']))
        df_train_y_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_y_processed']))
        
        x_df = df_train_x_features_tsfresh.sample(frac=1, random_state=42).copy()
        y_df = df_train_y_features.set_index('sku').loc[x_df.index].copy()
        
        x_manual_df = df_train_x_features.set_index('sku').loc[x_df.index].copy()
        x_manual_df = self._prepare_train_manual_features(x_manual_df)
        
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
        
        features = []
        with open('dataset/tsfresh_selected_features.txt', 'r', encoding='utf8') as f:
            for column in f:
                features.append(column[:-1])
                
        x_df = x_df[features]
        
        x_df_imputed = []
        with Pool(96) as p:
            for data in tqdm(p.imap(impute_pool_train, x_df.columns), total=len(x_df.columns)):
                x_df_imputed.append(data)

        x_df = pd.concat(x_df_imputed, axis=1) 
        
        x_df_filtered = x_df.loc[index_list]
        x_manual_df_filtered = x_manual_df.loc[index_list]
        
        X = np.concatenate((x_df_filtered.values, np.reshape(y_ts, (-1, 1))), axis=1)
        X = np.concatenate((X, x_manual_df_filtered.values), axis=1)
        y = y_td
        
        self.prepared_dataset = (X, y)
        self.features = features
        
    def train(self):
        X, y = self.prepared_dataset
        self.model = XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1,
                    random_state=0, tree_method='gpu_hist',
                    objective='multi:softprob', num_class=30, use_label_encoder=False,
                    gpu_id=3)
        self.model.fit(X, y)
    
    def predict(self, df_test):
        global x_test_df
        df_test = read_df(df_test)
        
        df_test_fromtrain_x_features = read_df(os.path.join(self.dataset_path, self.default_paths['test_fromtrain_data_x_processed']))   
        df_test_fromtrain_x_features_tsfresh = read_df(os.path.join(self.dataset_path, self.default_paths['test_fromtrain_data_x_processed_tsfresh']))
        
        x_test_df = df_test_fromtrain_x_features_tsfresh[self.features].loc[df_test['sku']].copy()
        
        x_test_manual_df = df_test_fromtrain_x_features.set_index('sku').loc[df_test['sku']].copy()
        X_test_manual = self._prepare_test_manual_features(x_test_manual_df).values
        
        x_test_df_imputed = []
        with Pool(96) as p:
            for data in tqdm(p.imap(impute_pool_test, x_test_df.columns), total=len(x_test_df.columns)):
                x_test_df_imputed.append(data)

        x_test_df = pd.concat(x_test_df_imputed, axis=1)
        
        X_test = np.concatenate((x_test_df.values, np.reshape(df_test['target_stock'].values, (-1, 1))), axis=1)
        X_test = np.concatenate((X_test, X_test_manual), axis=1)
        
        preds = self.model.predict_proba(X_test)
        probabilities = (preds/preds.sum(axis=1)[:,None]).round(4)
        
        return probabilities

