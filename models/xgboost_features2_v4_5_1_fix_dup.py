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
from sklearn.preprocessing import StandardScaler

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
    
    #df['target_stock__by_item_domain_id__target_day_by_mean'] = df['target_stock']/df['sold_quantity__by_item_domain_id__mean']
    #df['target_stock__by_item_domain_id__target_day_by_mean__std'] = df['sold_quantity__by_item_domain_id__std']*df['target_stock__by_item_domain_id__target_day_by_mean']/df['sold_quantity__by_item_domain_id__mean']
    
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for feature in df.columns:
        if feature.startswith('target_stock__'):
            df[feature] = impute(df[[feature]])[feature]
            
def normalize(df, scaler=None, normalize_positional=True, one_hot_categorical=False, one_hot_positional=False):
    categories_to_eliminate = ['item_domain_id', 'item_id', 'product_id', 'product_family_id']
    counting_to_eliminate = ['count']
    positional_to_eliminate = ['date__last_month', 'date__first_month', 'date__diff', 'date__last_weekofmonth', 'date__last_day', 'date__last_dayofweek']
    #numerical_to_eliminate = ['__sum', '']
    
    numeric_features = []
    categorical_features = []
    positional_features = []
    counting_features = []
    date_features = []
    string_features = []
    series_features = []
    id_features = []
    target_stock_features = []
    for feature in df.columns:
        feature_components = feature.split('__')
        if feature_components[-1] == 'target_stock':
            target_stock_features.append(feature)
        elif ('with' in feature_components[-1]) or ('location' in feature_components[-1]):
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
    
    numeric_features_tmp = []
    categorical_features_tmp = []
    positional_features_tmp = []
    counting_features_tmp = []
    #target_stock_features = ['target_stock']
    
    #for feature in df.columns:
    #    if feature.startswith('target_stock__'):
    #        df[feature] = df[feature]/30
    #        target_stock_features.append(feature)
    
    for feature in categorical_features:
        feature_components = feature.split('__')
        if feature_components[0] not in categories_to_eliminate:
            categorical_features_tmp.append(feature)
            if one_hot_categorical:
                pass #TODO implementar onehot
            else:
                df[feature] = df[feature].cat.codes
                
    for feature in counting_features:
        feature_components = feature.split('__')
        if feature_components[0] not in counting_to_eliminate:
            if 'mode' in feature_components[-1]:
                count_feature_base = df['count'] if feature_components[1] != 'by_item_domain_id' else df['count__by_item_domain_id']
            else:
                count_feature_base = "__".join(feature_components[:-1])
                count_feature_base = df[count_feature_base+'__count_of_zero'] + df[count_feature_base+'__count_of_non_zero']
            df[feature] = df[feature]/count_feature_base
            df[feature] = df[feature].fillna(0)
            counting_features_tmp.append(feature)
    
    for feature in positional_features:
        if feature in positional_to_eliminate:
            continue
        feature_components = feature.split('__')
        is_count_based = False
        if 'dayoftheweek' in feature_components[-2] or 'dayofweek' in feature_components[-1]:
            sub = 0
            positional_base = 7
        elif 'weekofthemonth' in feature_components[-2] or 'weekofmonth' in feature_components[-1]:
            sub = 0
            positional_base = 4
        elif 'dayofthemonth' in feature_components[-2] or 'day' in feature_components[-1]:
            sub = 1
            positional_base = 31
        else:
            is_count_based = True
            sub = 0
            positional_base = df['count']
            
        if normalize_positional:
            if one_hot_positional:
                pass #TODO implementar onehot
            else:
                df[feature] = (df[feature] - sub)/positional_base
                positional_features_tmp.append(feature)
        else:
            if is_count_based:
                df[feature] = df[feature].apply(lambda x: x if x >= 0 else np.nan)
                df[feature] = df[feature]-df['count']+29
                df[feature] = df[feature].fillna(-31)
            positional_features_tmp.append(feature)
            
    for feature in numeric_features:
        feature_components = feature.split('__')
        if 'linregress' in feature_components[-1] or 'energy_ratio' in feature_components[-1]:
            numeric_features_tmp.append(feature)
        elif 'variance_large_than_std' == feature_components[-1] or 'symmetry_looking' in feature_components[-1]:
            df[feature] = df[feature].astype(int)
            numeric_features_tmp.append(feature)
        elif feature_components[-1] == 'sum' or feature_components[-1] == 'abs_energy':
            #if 'by' not in feature_components[-2] and 'last' not in feature_components[-2]  and 'change' not in feature_components[-2]:
            #    df[feature] = df[feature]*29/df['count']    
            if 'change' in feature_components[-2]:
                df[feature] = df[feature]*28/(df['count']-1)
                df[feature] = df[feature].fillna(0)
            elif 'last' in feature_components[-2]:
                numeric_features_tmp.append(feature)
        else:
            numeric_features_tmp.append(feature)
    
    to_norm_features = numeric_features_tmp + target_stock_features
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[to_norm_features])
        
    df[to_norm_features] = scaler.transform(df[to_norm_features])
        
    features = numeric_features_tmp + categorical_features_tmp + positional_features_tmp + counting_features_tmp + target_stock_features
    return df[features], scaler

class XGBoostFeaturesV4_5_1(Model):
    model_name = 'xgboost_features2_v4_5_1_fix_dup'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):        
        df_train_x_featuresv2 = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_x_processedv2']))
        df_train_y_features = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_y_processed']))
        
        x_df = df_train_x_featuresv2.set_index('sku').sample(frac=1, random_state=42).copy()
        y_df = df_train_y_features.set_index('sku').loc[x_df.index].copy()
        
        features = []
        for feature in x_df.columns:
            if 'by_item_domain_id' in feature:
                continue
            if 'by_dayofthemonth' in feature or 'by_dayoftheweek' in feature or 'by_weekofthemonth' in feature:
                continue
            else:
                features.append(feature)
                
        x_df = x_df[features].copy()
          
        y_sold_quantity_series = np.array(list(y_df['sold_quantity_series'].apply(json.loads).values))
        y_cumsum = y_sold_quantity_series.cumsum(axis=1)

        SELECT_N = 3

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
        target_stock_features(x_df_train)
        x_df_train, scaler = normalize(x_df_train)
        
        print('Total features eliminated:', len(df_train_x_featuresv2.columns)-len(x_df_train.columns))
        
        X = x_df_train.values
        
        self.prepared_dataset = (X, y)
        self.features = features
        self.scaler = scaler
        
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
        
        x_test_df_all = df_train_featuresv2.set_index('sku').loc[df_test['sku']]
        x_test_df_all = x_test_df_all[self.features].copy()
        
        x_test_df_all['target_stock'] = df_test.set_index('sku')['target_stock']
        
        target_stock_features(x_test_df_all)
        x_test_df_all, scaler = normalize(x_test_df_all, scaler=self.scaler)
        X_test = x_test_df_all.values
        
        preds = self.model.predict_proba(X_test)
        probabilities = (preds/preds.sum(axis=1)[:,None]).round(4)
        
        return probabilities

