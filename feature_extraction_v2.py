import os
import sys
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import linregress

from utils import read_df, write_df, read_json

FEATURES_METADATA_PATH = './dataset/features_metadata.json'
features_metadata = read_json(FEATURES_METADATA_PATH)

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

def extract_series_default_features(new_row, series, name, zerout=False):
    if series.shape[0] == 0:
        zerout = True
    
    if not zerout:
        new_row[name+'__sum'] = series.sum()
        new_row[name+'__mean'] = series.mean()
        new_row[name+'__median'] = np.median(series)
        new_row[name+'__std'] = series.std()
        new_row[name+'__var'] = series.var()
        new_row[name+'__variance_large_than_std'] = new_row[name+'__var'] > new_row[name+'__std']
        new_row[name+'__min'] = series.min()
        new_row[name+'__max'] = series.max()
        new_row[name+'__abs_energy'] = np.dot(series, series)
        new_row[name+'__count_of_zero'] = np.count_nonzero(series==0)
        new_row[name+'__count_of_non_zero'] = np.count_nonzero(series)
    else:
        new_row[name+'__sum'] = 0
        new_row[name+'__mean'] = 0
        new_row[name+'__median'] = 0
        new_row[name+'__std'] = 0
        new_row[name+'__var'] = 0
        new_row[name+'__variance_large_than_std'] = False
        new_row[name+'__min'] = 0
        new_row[name+'__max'] = 0
        new_row[name+'__abs_energy'] = 0
        new_row[name+'__count_of_zero'] = 0
        new_row[name+'__count_of_non_zero'] = 0
        
def dayoftheweek_filter(df_dt, n):
    return df_dt.dayofweek == n

def weekofthemonth_filter(df_dt, n):
    return np.floor(df_dt.day/((df_dt.daysinmonth + 1)/4)) == n

def extract_series_date_features(new_row, df, name, date_filter, quantity):
    column = df.columns[-1]
    
    zeros_counts = []
    non_zeros_counts = []
    mean_values = []
    for n_date in range(quantity):
        n_date_prefix = name + '_' + str(n_date)
        df_date = df[date_filter(df['date'].dt, n_date)]
        
        series = df_date[column].values
        extract_series_default_features(new_row, series, n_date_prefix)

        zeros_counts.append(new_row[n_date_prefix+'__count_of_zero'])
        non_zeros_counts.append(new_row[n_date_prefix+'__count_of_non_zero'])
        mean_values.append(new_row[n_date_prefix+'__mean'])

    new_row[name+'__with_most_count_of_zero'] = np.argmax(zeros_counts)
    new_row[name+'__with_most_count_of_non_zero'] = np.argmax(non_zeros_counts)
    new_row[name+'__with_bigger_mean'] = np.argmax(mean_values)
    new_row[name+'__with_least_count_of_zero'] = np.argmin(zeros_counts)
    new_row[name+'__with_least_count_of_non_zero'] = np.argmin(non_zeros_counts)
    new_row[name+'__with_smaller_mean'] = np.argmin(mean_values)
    
def create_features_per_sku(df):
    new_row = df[item_columns + [id_column]].iloc[0].to_dict()

    count = len(df)
    new_row['count'] = count
    new_row['date__first'] = df[date_column].iloc[0]
    new_row['date__last'] = df[date_column].iloc[-1]
    new_row['date__diff'] = (new_row['date__last'] - new_row['date__first']).days
    for date in ['date__first', 'date__last']:
        new_row[date+'_day'] = new_row[date].day
        new_row[date+'_month'] = new_row[date].month
        new_row[date+'_dayofweek'] = new_row[date].dayofweek
        new_row[date+'_weekofmonth'] = np.floor(new_row[date].day/((new_row[date].daysinmonth + 1)/4)).astype(int)


    for column in sku_columns:
        new_row[column+'__first'] = df[column].iloc[0]
        new_row[column+'__last'] = df[column].iloc[-1]
        new_row[column+'__mode'] = df[column].mode().iloc[0]
        new_row[column+'__count_of_mode'] = df[column].value_counts().iloc[0]

    for column in sku_numeric_columns:
        series = df[column].values
        extract_series_default_features(new_row, series, column)
        new_row[column+'__last_location_of_maximum'] = count - 1 - np.argmax(series[::-1])
        new_row[column+'__last_location_of_minimum'] = count - 1 - np.argmin(series[::-1])
        zero_locations = np.where(series == 0)[0]
        non_zeros_locations = np.where(series != 0)[0]
        new_row[column+'__last_location_of_zero'] = zero_locations[-1] if zero_locations.shape[0] != 0 else -1 
        new_row[column+'__last_location_of_non_zero'] = non_zeros_locations[-1] if non_zeros_locations.shape[0] != 0 else -1

        #Boolean variable denoting if the distribution of x *looks symmetric*  | mean(X)-median(X)| < r * (max(X)-min(X))
        mean_minus_median_abs = np.abs(new_row[column+'__mean'] - new_row[column+'__median'])
        max_minus_min = new_row[column+'__max'] - new_row[column+'__min']
        for r in [0.3, 0.5, 0.7]:
            new_row[column+f'__symmetry_looking_r_{r}'] = mean_minus_median_abs < r*max_minus_min

        #series energy ratio last chunk
        full_series_energy = np.sum(series ** 2)
        for n_chuncks, focus in [(3,2), (5,4), (10,9)]:
            if full_series_energy != 0:
                series_split = np.array_split(series, n_chuncks)
                selected_series = series_split[focus]
                for i in range(focus, -1, -1):
                    if series_split[i].shape[0] > 0:
                        selected_series = series_split[i]
                        break
                new_row[column+f'__energy_ratio_by_chunks_{n_chuncks}_segments_focus_{focus}'] = np.sum(selected_series ** 2) / full_series_energy
            else:
                new_row[column+f'__energy_ratio_by_chunks_{n_chuncks}_segments_focus_{focus}'] = 0

        if count > 1:
            regression = linregress(np.arange(count), series)
            new_row[column+'__linregress_slope'] = regression.slope
            new_row[column+'__linregress_intercept'] = regression.intercept
            new_row[column+'__linregress_pvalue'] = regression.pvalue
            new_row[column+'__linregress_rvalue'] = regression.rvalue
            new_row[column+'__linregress_stderr'] = regression.stderr
        else:
            new_row[column+'__linregress_slope'] = 0
            new_row[column+'__linregress_intercept'] = series[0]
            new_row[column+'__linregress_pvalue'] = 0
            new_row[column+'__linregress_rvalue'] = 0
            new_row[column+'__linregress_stderr'] = 0

        series_change = np.diff(series)
        extract_series_default_features(new_row, series_change, column+'__change')

        series_change_abs = np.abs(series_change)
        extract_series_default_features(new_row, series_change_abs, column+'__change_absolute')

        last_5_series = series[-5:]
        extract_series_default_features(new_row, last_5_series, column+'__last_5')

        last_3_series = series[-3:]
        extract_series_default_features(new_row, last_5_series, column+'__last_3')

        weekdate_prefix = '__by_dayoftheweek'
        extract_series_date_features(new_row, df[[date_column, column]], column+weekdate_prefix, dayoftheweek_filter, 7)

        monthweek_prefix = '__by_weekofthemonth'
        extract_series_date_features(new_row, df[[date_column, column]], column+monthweek_prefix, weekofthemonth_filter, 4)

    for column in sku_columns:
        new_row[column+'__series'] = df[column].to_json(orient='values')
    
    return new_row

def create_features_per_item_domain_id(df):
    df_group_date = df.groupby(date_column)

    count = len(df)
    new_row = {
        'item_domain_id': df['item_domain_id'].iloc[0],
        'count__by_item_domain_id': count,
        'count_sku__by_item_domain_id': df[id_column].nunique()
    }

    by_prefix = '__by_item_domain_id'

    for column in sku_columns:
        new_row[column+by_prefix+'__mode'] = df[column].mode().iloc[0]
        new_row[column+by_prefix+'__count_of_mode'] = df[column].value_counts().iloc[0]

    for column in sku_numeric_columns:
        series = df[column].values
        extract_series_default_features(new_row, series, column+by_prefix)

        series = df_group_date[column].mean().values
        if series.shape[0] > 1:
            regression = linregress(np.arange(series.shape[0]), series)
            new_row[column+by_prefix+'__mean__linregress_slope'] = regression.slope
            new_row[column+by_prefix+'__mean__linregress_intercept'] = regression.intercept
            new_row[column+by_prefix+'__mean__linregress_pvalue'] = regression.pvalue
            new_row[column+by_prefix+'__mean__linregress_rvalue'] = regression.rvalue
            new_row[column+by_prefix+'__mean__linregress_stderr'] = regression.stderr
        else:
            new_row[column+by_prefix+'__mean__linregress_slope'] = 0
            new_row[column+by_prefix+'__mean__linregress_intercept'] = series[0]
            new_row[column+by_prefix+'__mean__linregress_pvalue'] = 0
            new_row[column+by_prefix+'__mean__linregress_rvalue'] = 0
            new_row[column+by_prefix+'__mean__linregress_stderr'] = 0

        weekdate_prefix = '__by_dayoftheweek'
        extract_series_date_features(new_row, df[[date_column, column]], column+by_prefix+weekdate_prefix, dayoftheweek_filter, 7)

        monthweek_prefix = '__by_weekofthemonth'
        extract_series_date_features(new_row, df[[date_column, column]], column+by_prefix+monthweek_prefix, weekofthemonth_filter, 4)

        daymonth_prefix = '__by_dayofthemonth'
        #new_row[column+by_prefix+daymonth_prefix+'__with_smaller_mean'] = df_group_date[column].max().idxmax().day
        count_zero = df_group_date[column].apply(lambda x: np.where(x == 0)[0].shape[0])
        count_non_zero = df_group_date[column].apply(lambda x: np.where(x != 0)[0].shape[0])
        new_row[column+by_prefix+daymonth_prefix+'__with_most_count_of_zero'] = count_zero.idxmax().day
        new_row[column+by_prefix+daymonth_prefix+'__with_most_count_of_non_zero'] = count_non_zero.idxmax().day
        new_row[column+by_prefix+daymonth_prefix+'__with_least_count_of_zero'] = count_zero.idxmin().day
        new_row[column+by_prefix+daymonth_prefix+'__with_least_count_of_non_zero'] = count_non_zero.idxmin().day

        count_date = df_group_date[column].count()
        new_row[column+by_prefix+daymonth_prefix+'__bigger_sum'] = df_group_date[column].sum().max()
        new_row[column+by_prefix+daymonth_prefix+'__bigger_mean'] = df_group_date[column].mean().max()
        new_row[column+by_prefix+daymonth_prefix+'__smaller_sum'] = df_group_date[column].sum().min()
        new_row[column+by_prefix+daymonth_prefix+'__smaller_mean'] = df_group_date[column].mean().min()
        new_row[column+by_prefix+daymonth_prefix+'__with_bigger_mean'] = df_group_date[column].sum().idxmax().day
        new_row[column+by_prefix+daymonth_prefix+'__with_bigger_mean'] = df_group_date[column].mean().idxmax().day
        new_row[column+by_prefix+daymonth_prefix+'__with_smaller_sum'] = df_group_date[column].mean().idxmin().day
        new_row[column+by_prefix+daymonth_prefix+'__with_smaller_mean'] = df_group_date[column].mean().idxmin().day

        if count_date.shape[0] > 1 and np.any(count_date > 1):
            new_row[column+by_prefix+daymonth_prefix+'__bigger_std'] = df_group_date[column].std().max()
            new_row[column+by_prefix+daymonth_prefix+'__bigger_var'] = df_group_date[column].var().max()
            new_row[column+by_prefix+daymonth_prefix+'__smaller_std'] = df_group_date[column].std().min()
            new_row[column+by_prefix+daymonth_prefix+'__smaller_var'] = df_group_date[column].var().min()
            new_row[column+by_prefix+daymonth_prefix+'__with_bigger_std'] = df_group_date[column].std().idxmax().day
            new_row[column+by_prefix+daymonth_prefix+'__with_bigger_var'] = df_group_date[column].var().idxmax().day
            new_row[column+by_prefix+daymonth_prefix+'__with_smaller_std'] = df_group_date[column].std().idxmin().day
            new_row[column+by_prefix+daymonth_prefix+'__with_smaller_var'] = df_group_date[column].var().idxmin().day
        else:
            new_row[column+by_prefix+daymonth_prefix+'__bigger_std'] = 0
            new_row[column+by_prefix+daymonth_prefix+'__bigger_var'] = 0
            new_row[column+by_prefix+daymonth_prefix+'__smaller_std'] = 0
            new_row[column+by_prefix+daymonth_prefix+'__smaller_var'] = 0
            new_row[column+by_prefix+daymonth_prefix+'__with_bigger_std'] = df_group_date[column].std().index[0].day
            new_row[column+by_prefix+daymonth_prefix+'__with_bigger_var'] = df_group_date[column].var().index[0].day
            new_row[column+by_prefix+daymonth_prefix+'__with_smaller_std'] = df_group_date[column].std().index[0].day
            new_row[column+by_prefix+daymonth_prefix+'__with_smaller_var'] = df_group_date[column].var().index[0].day

    return new_row

def extract_features_per_sku(data_train, data_items, n_workers=100):
    start_time = time.time()
    df_train = read_df(data_train)
    df_item = read_df(data_items)
    
    df_all = df_train.merge(df_item, on='sku')
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    for column in categorical_columns: 
        df_all[column] = df_all[column].astype(str)
        
    df_all = df_all.sort_values(['item_domain_id', 'sku', 'date']).reset_index(drop=True)
    
    print("preprocess_data: " + str(time.time() - start_time) + " seconds")
    
    df = df_all   
    df_first_entry = df[df['sku'].diff() != 0]
    sku_split = []
    for i in range(len(df_first_entry)-1):
        start_index = df_first_entry.index[i]
        end_index = df_first_entry.index[i+1]
        sku_split.append(df.iloc[start_index:end_index])
    sku_split.append(df.iloc[end_index:].copy().reset_index(drop=True))
    
    print("sku_split: " + str(time.time() - start_time) + " seconds")
    
    sku_data = []
    with Pool(n_workers) as p:
        for data in tqdm(p.imap(create_features_per_sku, sku_split), total=len(sku_split)):
            sku_data.append(data)
            
    print("sku_processing: " + str(time.time() - start_time) + " seconds")
            
    df_sku = pd.DataFrame(sku_data)
    
    df = df_all['item_domain_id'].copy()
    df = df.astype('category').cat.codes
    df_first_entry = df[df.diff() != 0]
    indexes = []
    for i in range(len(df_first_entry)-1):
        start_index = df_first_entry.index[i]
        end_index = df_first_entry.index[i+1]
        indexes.append((start_index, end_index))
    
    del df
    
    item_domain_id_split = []
    for index in indexes:
        item_domain_id_split.append(df_all.iloc[index[0]:index[1]].copy().reset_index(drop=True))
    item_domain_id_split.append(df_all.iloc[index[1]:].copy().reset_index(drop=True))
    
    print("item_domain_split: " + str(time.time() - start_time) + " seconds")
    
    #item_domain_id_split =[]
    #for item_domain_id in df_sku['item_domain_id'].unique():
    #    df = df_all[df_all['item_domain_id'] == item_domain_id]
    #    item_domain_id_split.append(df.copy().reset_index(drop=True))
        
    item_domain_id_data = []
    with Pool(n_workers) as p:
        for data in tqdm(p.imap(create_features_per_item_domain_id, item_domain_id_split), total=len(item_domain_id_split)):
            item_domain_id_data.append(data)
            
    print("item_domain_processing: " + str(time.time() - start_time) + " seconds")
            
    df_item_domain = pd.DataFrame(item_domain_id_data)
    
    df_features_v2 = df_sku.merge(df_item_domain, on='item_domain_id')
    
    #####################
    # Fix dtypes
    #####################
    numeric_features = []
    categorical_features = []
    positional_features = []
    counting_features = []
    date_features = []
    string_features = []
    series_features = []
    id_features = []
    for feature in df_features_v2.columns:
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
            
    for feature in categorical_features:
        feature_components = feature.split('__')
        feature_base = feature_components[0]
        df_features_v2[feature] = df_features_v2[feature].astype(str).astype('category')
        df_features_v2[feature] = df_features_v2[feature].cat.set_categories(features_metadata['train']['features'][feature_base]['categories'])
    
    for feature in string_features:
        df_sku[feature] = df_sku[feature].astype(str)
        
    for feature in date_features:
        df_features_v2[feature] = pd.to_datetime(df_features_v2[feature])
    
    for feature in series_features:
        df_features_v2[feature] = df_features_v2[feature].astype(str)
    
    print("Process fineshed after: " + str(time.time() - start_time) + " seconds")
    return df_features_v2
    
if __name__ == "__main__":
    data_train = sys.argv[-3]
    data_items = sys.argv[-2]
    out_path = sys.argv[-1]
    print('data_train:', data_train)
    print('data_items:', data_items)
    print('out_path:', out_path)
    print('Extracting features v2...')
    df_sku = extract_features_per_sku(data_train, data_items)
    print(f'Saving DataFrame on {out_path}')
    write_df(df_sku, out_path)
    print('Done')