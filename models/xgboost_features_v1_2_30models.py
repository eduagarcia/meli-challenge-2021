import numpy as np
import json
import os
import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool
from scipy.stats import norm
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=50)

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

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
        
series_columns = ['sold_quantity_series', 'current_price_series', 'minutes_active_series']
series_columns_headers = {}

series_columns_max = 29
for series_name in series_columns:
    series_columns_headers[series_name] = [series_name+'_'+str(i) for i in range(series_columns_max)]
    numeric_columns = numeric_columns + series_columns_headers[series_name]

features = numeric_columns + categorical_columns

general_mean = 1.5
general_std = 2

def normal_probs(data):
    row, pred = data
    sku = row['sku']
    target_stock = row['target_stock']
    
    sold_quantity_series = pred
    
    sold_mean = sold_quantity_series.mean()
    sold_std  = sold_quantity_series.std()

    if sold_mean == 0:
        sold_mean = general_mean
        sold_std = general_std
    if sold_std == 0:
        sold_std = general_std

    days_stockout = target_stock/sold_mean
    std_days = (sold_std / sold_mean) * days_stockout

    dist_model = norm(days_stockout,std_days)

    probalities = np.zeros(30)
    for i in range(1, 31):
        probalities[i-1] = (dist_model.cdf(i+1) - dist_model.cdf(i))

    if probalities.sum() == 0:
        probalities = np.ones(30) / 30

    probalities = (probalities/probalities.sum()).round(4)
    #probalities = saferound(probalities, places=4)
    return (sku, probalities)

class XGBoostFeaturesV1_2(Model):
    model_name = 'xgboost_features_v1_1_30models'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        print('Number of features:', len(features), 'Non-used features: ', set(all_columns) - set(features) - set(series_columns))
        
        df_train_x_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_x_processed']))
        df_train_y_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_y_processed']))
        
        #Randomize and order sync train data
        x_df = df_train_x_features.sample(frac=1, random_state=42).reset_index(drop=True).copy()
        y_df = df_train_y_features.set_index('sku').loc[x_df['sku']].reset_index().copy()
        
        for column in numeric_columns_composed:
            x_df[column+'_std'] = x_df[column+'_std'].fillna(0)
            y_df[column+'_std'] = y_df[column+'_std'].fillna(0)
        for column in categorical_columns_with_nan:    
            x_df[column] = x_df[column].astype(str).astype("category")
            y_df[column] = y_df[column].astype(str).astype("category")
            
        categorical_pandas_order = {}

        for column in categorical_columns:    
            x_df[column] = x_df[column].astype("category")
            categorical_pandas_order[column] = x_df[column].cat.categories
            y_df[column] = y_df[column].astype("category")
            y_df[column] = y_df[column].cat.set_categories(categorical_pandas_order[column], ordered=True)
        
        for column in categorical_columns:
            x_df[column] = x_df[column].cat.codes
            y_df[column] = y_df[column].cat.codes
            
        for column in series_columns:
            def series_to_df(data):
                result = np.ones(series_columns_max)*-1
                data = json.loads(data)
                index_limit = len(data) if len(data) <= series_columns_max else series_columns_max
                result[-index_limit:] = data[-index_limit:]
                #return result
                return pd.Series(result, index=series_columns_headers[column])

            series_x_df = x_df[column].parallel_apply(series_to_df)
            series_y_df = y_df[column].parallel_apply(series_to_df)

            x_df = x_df.join(series_x_df)
            y_df = y_df.join(series_y_df)
        
        X = x_df[features].values
        y = np.array(list(y_df['sold_quantity_series'].apply(json.loads).values))
        #y = y.cumsum(axis=1)
        
        self.prepared_dataset = (X, y)
        self.categorical_pandas_order = categorical_pandas_order
    
    def train(self):
        X, y = self.prepared_dataset
        self.model = MultiOutputRegressor(XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.1,
                    random_state=0, tree_method='gpu_hist',
                    gpu_id=3), n_jobs=8)
        self.model.fit(X, y)
    
    def predict(self, df_test):
        df_test = read_df(df_test)
        df_test_fromtrain_x_features = read_df(os.path.join(self.dataset_path, self.default_paths['test_fromtrain_data_x_processed']))
        
        test_x_df = df_test_fromtrain_x_features.set_index('sku').loc[df_test['sku']].reset_index().copy()
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
        for column in  series_columns:
            def series_to_df(data):
                result = np.ones(series_columns_max)*-1
                data = json.loads(data)
                index_limit = len(data) if len(data) <= series_columns_max else series_columns_max
                result[-index_limit:] = data[-index_limit:]
                #return result
                return pd.Series(result, index=series_columns_headers[column])

            series_x_test_df = test_x_df[column].parallel_apply(series_to_df)
            test_x_df = test_x_df.join(series_x_test_df)

        X_test = test_x_df[features].values
        #X_test = np.concatenate((X_test, np.reshape(df_test['target_stock'].values, (-1, 1))), axis=1)
        
        preds = self.model.predict(X_test)
        
        skus = []
        probabilities = np.zeros((len(df_test), 30))
        i = 0
        with Pool(100) as p:
            for data in tqdm(p.imap(normal_probs, zip(df_test.to_dict(orient='records'), preds)), total=len(df_test)):
                sku, probs = data
                skus.append(sku)
                probabilities[i] = probs
                i += 1
        
        skus = np.array(skus)
        comparison = skus == df_test['sku'].to_numpy()
        assert comparison.all()
        
        return probabilities

