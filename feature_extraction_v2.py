import os
import sys
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import read_df, write_df

df_item = None
    
def create_features_per_sku(df_sku):
    df = df_sku
    sku = df['sku'].iloc[0]
    new_row = df_item.loc[sku].to_dict()
    
    df['datetime'] = pd.to_datetime(df['date'])

    new_row['count'] = len(df)
    new_row['date_first'] = df['datetime'].iloc[0]
    new_row['date_last'] = df['datetime'].iloc[-1]
    new_row['date_diff'] = (new_row['date_last'] - new_row['date_first']).days
    for date in ['date_first', 'date_last']:
        new_row[date+'_day'] = new_row[date].day
        new_row[date+'_month'] = new_row[date].month

    for quantity in ['sold_quantity', 'current_price', 'minutes_active']:
        new_row[quantity+'_first'] = df[quantity].iloc[0]
        new_row[quantity+'_last'] = df[quantity].iloc[-1]
        new_row[quantity+'_sum'] = df[quantity].sum()
        new_row[quantity+'_mean'] = df[quantity].mean()
        new_row[quantity+'_std'] = df[quantity].std()
        new_row[quantity+'_min'] = df[quantity].min()
        new_row[quantity+'_max'] = df[quantity].max()
        new_row[quantity+'_mode'] = df[quantity].mode().iloc[0]
        new_row[quantity+'_mode_tx'] = df[quantity].value_counts().iloc[0]/new_row['count']

    for category in ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']:
        new_row[category+'_first'] = df[category].iloc[0]
        new_row[category+'_last'] = df[category].iloc[-1]
        new_row[category+'_mode'] = df[category].mode().iloc[0]
        new_row[category+'_mode_tx'] = df[category].value_counts().iloc[0]/new_row['count']

    new_row['minutes_active_series'] = df['sold_quantity'].to_json(orient='values')
    new_row['current_price_series'] = df['sold_quantity'].to_json(orient='values')
    new_row['sold_quantity_series'] = df['sold_quantity'].to_json(orient='values')
    return new_row

def extract_features_per_sku(data_train, data_items, n_workers=100):
    global df_item
    df = read_df(data_train)
    df_item = read_df(data_items)
    
    df = df.sort_values(['sku', 'date']).reset_index(drop=True)
    df_first_entry = df[df['sku'].diff() != 0]
    
    sku_split = []
    for i in range(len(df_first_entry)-1):
        start_index = df_first_entry.index[i]
        end_index = df_first_entry.index[i+1]
        sku_split.append(df.iloc[start_index:end_index])
    sku_split.append(df.iloc[end_index:])
    
    sku_data = []
    with Pool(n_workers) as p:
        for data in tqdm(p.imap(create_features_per_sku, sku_split), total=len(sku_split)):
            sku_data.append(data)
            
    df_sku = pd.DataFrame(sku_data)
    
    #fix dtypes
    for category in ['currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment']:
        df_sku[category+'_first'] = df_sku[category+'_first'].astype("category")
        df_sku[category+'_last'] = df_sku[category+'_last'].astype("category")
        df_sku[category+'_mode'] = df_sku[category+'_mode'].astype("category")
        
    for category in ['item_domain_id', 'item_id', 'site_id', 'product_id', 'product_family_id']:
        df_sku[category] = df_sku[category].astype("category")
    
    df_sku['item_title'] = df_sku['item_title'].astype(str)
    df_sku['minutes_active_series'] = df_sku['minutes_active_series'].astype(str)
    df_sku['current_price_series'] = df_sku['current_price_series'].astype(str)
    df_sku['sold_quantity_series'] = df_sku['sold_quantity_series'].astype(str)
    
    return df_sku
    
if __name__ == "__main__":
    data_train = sys.argv[-3]
    data_items = sys.argv[-2]
    out_path = sys.argv[-1]
    print('data_train:', data_train)
    print('data_items:', data_items)
    print('out_path:', out_path)
    print('Extracting features...')
    df_sku = extract_features_per_sku(data_train, data_items)
    print(f'Saving DataFrame on {out_path}')
    write_df(df_sku, out_path)
    print('Done')