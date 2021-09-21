#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pandas as pd
import datetime
import numpy as np
from fastparquet import write
import sklearn
from sklearn.model_selection import KFold
from utils import read_df, write_df
from feature_extraction import extract_features_per_sku


# In[2]:


DATASET_FOLDER = "./dataset"
#train_processed_data = os.path.join(DATASET_FOLDER, 'processed/sku_feature_data.parquet')
train_data = os.path.join(DATASET_FOLDER, 'train_data.parquet')
item_data = os.path.join(DATASET_FOLDER, 'items_static_metadata_full.jl')


# In[3]:


target_folder = "./dataset/processed/train_v2"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)


# In[4]:


df_train = pd.read_parquet(train_data, engine='fastparquet')

#df_train_v1 = pd.read_parquet(train_processed_data, engine='fastparquet')
#df_train_v1 = df_train_v1.set_index('sku')

df_item = pd.read_json(item_data, lines=True)


# In[5]:


skus = df_train['sku'].unique()


# In[6]:


# Create sku to index on train data
index_to_sku = df_train[df_train['sku'].diff() != 0]['sku']
shifted_index = np.append(index_to_sku.index.values[1:].copy(), [len(df_train)])
index_range = list(zip(index_to_sku.index.values, shifted_index))
sku_to_index_range = pd.Series(index_range, index=index_to_sku)


# In[7]:


kf = KFold(n_splits=4, random_state=None, shuffle=True)


# In[8]:


for kfold, (train_index, test_index) in enumerate(kf.split(skus)):
    print(kfold, len(train_index), len(test_index))
    
    #Create folder structure
    data_target_folder = os.path.join(target_folder, str(kfold))
    if not os.path.exists(data_target_folder):
        os.makedirs(data_target_folder)
    
    #Pick last 30 datapoints from train
    def pick_last_30(index_range):
        x1, x2 = index_range
        if x2-x1 <= 30:
            return np.array([np.nan])
        else:
            return np.arange(x2-30, x2)

    test_df_index = np.concatenate(sku_to_index_range[test_index].apply(pick_last_30).values)
    test_df_index = test_df_index[~np.isnan(test_df_index)].astype('int64')
    df_kfold_test = df_train.loc[test_df_index]
    
    #Remove sku's with total sold_quantity == 0 from test set
    test_zero_solded_sku = df_kfold_test.groupby('sku')['sold_quantity'].sum() == 0
    invalid_sku = test_zero_solded_sku[test_zero_solded_sku].index.values
    df_kfold_test = df_kfold_test[~df_kfold_test['sku'].isin(invalid_sku)]
    
    #Rebuild train data with rows not in test data
    df_kfold_train = df_train.loc[~df_train.index.isin(df_kfold_test.index)].copy().reset_index(drop=True)
    df_kfold_test = df_kfold_test.copy().reset_index(drop=True)
    
    #Build test data with random target_stock
    test_sold_quantity_agg = df_kfold_test.groupby('sku')['sold_quantity'].agg(list).apply(np.array)
    test_sold_quantity_possibilites = test_sold_quantity_agg.apply(np.cumsum).apply(lambda x: np.unique(x, return_index=True))
    
    def random_choose_target_stock(x):
        target_stocks, target_dates = x
        #ignore target_stock == 0
        if target_stocks[0] == 0:
            target_stocks = target_stocks[1:]
            target_dates = target_dates[1:]
        randint = np.random.randint(0, target_stocks.shape[0])
        target_date = target_dates[randint]
        target_stock = target_stocks[randint]
        return target_stock, target_date_0

    test_data = test_sold_quantity_possibilites.apply(random_choose_target_stock)
    test_data = pd.DataFrame([[sku, stock, date] for sku, (stock, date) in zip(test_data.index.values, test_data.values)], columns=['sku','target_stock', 'target_date_0'])
    
    ground_truth = np.eye(30)[test_data['target_date_0'].values]
    
    #Write data to folder
    write_df(df_kfold_train, os.path.join(data_target_folder, 'train_data.parquet'))
    write_df(df_kfold_test, os.path.join(data_target_folder, 'test_fromtrain_data.parquet'))
    write_df(test_data[['sku', 'target_stock']], os.path.join(data_target_folder, 'test_data.csv'))
    write_df(pd.DataFrame(ground_truth), os.path.join(data_target_folder, 'test_ground_truth.csv'), header=False)
    np.save(os.path.join(data_target_folder, 'test_ground_truth.npy'), ground_truth)
    
    with open(os.path.join(data_target_folder, 'test_sku.txt'), 'w') as f:
        for sku in test_data['sku']:
            f.write(str(sku)+'\n')
            
    #Feature extraction train data
    df_kfold_train_processed = extract_features_per_sku(df_kfold_train, df_item)
    
    write_df(df_kfold_train_processed, os.path.join(data_target_folder, 'train_sku_feature_data.parquet'))

