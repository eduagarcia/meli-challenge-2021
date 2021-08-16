import os
import sys
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from utils import read_df, write_df

def extract_features_tsfresh(data_train, n_workers=96):
    global df_item
    df = read_df(data_train)
    
    df['date'] = pd.to_datetime(df['date'])
    
    id_column = 'sku'
    sort_column = 'date'
    numeric_columns = ['sold_quantity', 'current_price', 'minutes_active']

    features = [id_column] + [sort_column] + numeric_columns
    
    df = df[features].copy()
    
    extraction_settings = ComprehensiveFCParameters()
    
    df_features = extract_features(df, column_id=id_column, column_sort=sort_column,
                         default_fc_parameters=extraction_settings, n_jobs=n_workers, disable_progressbar=False)
    
    return df_features
    
if __name__ == "__main__":
    data_train = sys.argv[-2]
    out_path = sys.argv[-1]
    print('data_train:', data_train)
    print('out_path:', out_path)
    print('Extracting tsfresh features...')
    df_sku = extract_features_tsfresh(data_train)
    print(f'Saving DataFrame on {out_path}')
    write_df(df_sku, out_path)
    print('Done')