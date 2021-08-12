import os
import sys
import json

import numpy as np
import pandas as pd

from utils import read_df, write_df

def create_last_29_predict_data(data_train, data_test):
    df_train = read_df(data_train)
    df_test = read_df(data_test)
    
    training_data = df_train[df_train['sku'].isin(df_test['sku'])].reset_index(drop=True).copy()
    training_data['date'] = pd.to_datetime(training_data['date'])
    
    training_data = training_data.sort_values(['sku', 'date']).reset_index(drop=True)
    
    index_to_sku = training_data[training_data['sku'].diff() != 0]['sku']
    shifted_index = np.append(index_to_sku.index.values[1:].copy(), [len(training_data)])
    index_range = list(zip(index_to_sku.index.values, shifted_index))
    sku_to_index_range = pd.Series(index_range, index=index_to_sku)
    
    def pick_last_29(index_range):
        x1, x2 = index_range
        if x1-x2 > 29:
            return np.arange(x2-29, x2)
        else:
            return np.arange(x1, x2)

    target_data_indexes = np.concatenate(sku_to_index_range.apply(pick_last_29).values)
    last_29_df = training_data.loc[target_data_indexes].sort_values(['sku', 'date']).reset_index(drop=True).copy()
    
    return last_29_df
    
if __name__ == "__main__":
    data_train = sys.argv[-3]
    data_test = sys.argv[-2]
    out_path = sys.argv[-1]
    print('data_train:', data_train)
    print('data_test:', data_test)
    print('out_path:', out_path)
    print('Extracting last 29 data...')
    last_29_df = create_last_29_predict_data(data_train, data_test)
    print(f'Saving DataFrame on {out_path}')
    write_df(last_29_df, out_path)
    print('Done')