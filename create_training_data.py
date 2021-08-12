import os
import sys
import json

import numpy as np
import pandas as pd

from utils import read_df, write_df

def create_training_data(data_train):
    df_train = read_df(data_train)
    
    training_data_counts = df_train.groupby('sku')['sold_quantity'].count()
    training_data_counts = training_data_counts[training_data_counts > 30]
    training_data_skus = training_data_counts.index
    
    training_data = df_train[df_train['sku'].isin(training_data_skus)].copy()
    training_data['date'] = pd.to_datetime(training_data['date'])
    
    training_data = training_data.sort_values(['sku', 'date']).reset_index(drop=True)
    
    index_to_sku = training_data[training_data['sku'].diff() != 0]['sku']
    shifted_index = np.append(index_to_sku.index.values[1:].copy(), [len(training_data)])
    index_range = list(zip(index_to_sku.index.values, shifted_index))
    sku_to_index_range = pd.Series(index_range, index=index_to_sku)
    
    def pick_last_30(index_range):
        x1, x2 = index_range
        return np.arange(x2-30, x2)

    target_data_indexes = np.concatenate(sku_to_index_range.apply(pick_last_30).values)
    x_df = training_data.loc[~training_data.index.isin(target_data_indexes)].sort_values(['sku', 'date']).reset_index(drop=True).copy()
    y_df = training_data.loc[target_data_indexes].sort_values(['sku', 'date']).reset_index(drop=True).copy()
    
    return x_df, y_df
    
if __name__ == "__main__":
    data_train = sys.argv[-2]
    out_path = sys.argv[-1]
    print('data_train:', data_train)
    print('out_path:', out_path)
    print('Extracting train data...')
    x_df, y_df = create_training_data(data_train)
    print(f'Saving DataFrame on {out_path}')
    write_df(x_df, os.path.join(out_path, 'train_data_x.parquet'))
    write_df(y_df, os.path.join(out_path, 'train_data_y.parquet'))
    print('Done')