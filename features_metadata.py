import os
import sys
import json

import numpy as np
import pandas as pd

from utils import read_df, write_df

id_column =  'sku'
date_column = 'date'

item_string_columns = ['item_title']
item_categorical_columns = ['item_domain_id', 'item_id', 'site_id', 'product_id', 'product_family_id']
item_columns = item_string_columns + item_categorical_columns

sku_numeric_columns = ['sold_quantity', 'current_price', 'minutes_active']
sku_categorical_columns = ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']
sku_columns = sku_numeric_columns + sku_categorical_columns

test_columns = ['target_stock']

string_columns = item_string_columns
categorical_columns = sku_categorical_columns + item_categorical_columns
numeric_columns = sku_numeric_columns

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def extract_features_metadata(data_train, data_item, data_test):
    df_train = read_df(data_train)
    df_item = read_df(data_item)
    df_test = read_df(data_test)
    
    df_all = df_train.merge(df_item, on='sku')
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    features_data = {}

    for column in categorical_columns:
        data = {
            'column': column,
            'type': 'category',
            'dtype': str(df_all[column].dtype)
        }
        df_all[column] = df_all[column].astype(str).astype('category')
        value_counts = df_all[column].value_counts()
        data['categories'] = list(value_counts.index)
        data['value_counts'] = list(value_counts.values)
        #df_all[column].cat.set_categories(data['categories'])
        data['size'] = len(data['categories'])
        features_data[column] = data
        
    for column in numeric_columns:
        data = {
            'column': column,
            'type': 'numeric',
            'dtype': str(df_all[column].dtype)
        }
        data['min'] = df_all[column].min()
        data['max'] = df_all[column].max()
        data['var'] = df_all[column].var()
        data['std'] = df_all[column].std()
        data['mean'] = df_all[column].mean()
        data['sum'] = df_all[column].sum()
        value_counts = df_all[column].value_counts()
        data['unique'] = list(value_counts.index)
        data['value_counts'] = list(value_counts.values)
        data['size_unique'] = len(data['unique'])
        features_data[column] = data

    for column in string_columns:
        features_data[column] = {
            'column': column,
            'type': 'string',
            'dtype': 'str',
            'size_unique': df_all[date_column].nunique()
        }
        
    for column in item_columns:
        features_data[column]['dataset_type']: 'timeseries'
    for column in sku_columns:
        features_data[column]['dataset_type']: 'metadata'

    features_data[id_column] = {
        'column': id_column,
        'type': 'id',
        'dtype': str(df_all['sku'].dtype),
        'dataset_type': 'metadata',
        'size_unique': df_all['sku'].nunique()
    }

    features_data[date_column] = {
        'column': date_column,
        'type': 'date',
        'dataset_type': 'timeseries',
        'date_max': str(df_all[date_column].min()),
        'date_min': str(df_all[date_column].max()),
        'size_unique': df_all[date_column].nunique(),
    }
            
    features_test = {}
    for column in test_columns:
        data = {
            'column': column,
            'type': 'numeric',
            'dataset_type': 'test',
            'dtype': str(df_test[column].dtype)
        }
        data['min'] = df_test[column].min()
        data['max'] = df_test[column].max()
        data['var'] = df_test[column].var()
        data['std'] = df_test[column].std()
        data['mean'] = df_test[column].mean()
        data['sum'] = df_test[column].sum()
        value_counts = df_test[column].value_counts()
        data['unique'] = list(value_counts.index)
        data['value_counts'] = list(value_counts.values)
        data['size_unique'] = len(data['unique'])
        features_test[column] = data
        
    features_test[id_column] = {
        'column': id_column,
        'type': 'id',
        'dtype': str(df_test['sku'].dtype),
        'dataset_type': 'test',
        'size_unique': df_test['sku'].nunique()
    }
    
    return {
                'train': {
                    'features': features_data,
                    'size': len(df_train),
                    'size_meta': len(df_item),
                },
                'test': {
                    'features': features_test,
                    'size': len(df_test)
                }
           }
    
if __name__ == "__main__":
    data_train = sys.argv[-4]
    data_item = sys.argv[-3]
    data_test = sys.argv[-2]
    out_path = sys.argv[-1]
    print('data_train:', data_train)
    print('data_item:', data_item)
    print('data_test:', data_test)
    print('out_path:', out_path)
    print('Extracting features metadata...')
    result = extract_features_metadata(data_train, data_item, data_test)
    print(f'Saving DataFrame on {out_path}')
    with open(out_path, 'w') as f:
        json.dump(result, f, cls=NpEncoder)
    print('Done')