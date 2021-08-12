import numpy as np
import json
import os
import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool
from iteround import saferound
import scipy.stats as st
import tweedie
from category_encoders import OrdinalEncoder

from xgboost import XGBRegressor

from model import Model
from utils import read_df

DEFAULT_PARAMS = [0.003936128001463711, 2, 0.29539066512210194, 0.47989860558921493, 1.8040470414877383, 145]

CATEGORICAL_FEATURES = ['item_domain_id', 'currency', 'listing_type', 'shipping_logistic_type', 'shipping_payment', 'site_id']
FEATURES = ["current_price", "minutes_active"] + CATEGORICAL_FEATURES

def pred_list_to_tweedie(pred_list, phi=1, p=1.5):
    # has a bug in the first day, it's the wrong probability, but it's worse without the bug
    distros = dict()
    for mu in range(1,31):
        distros[mu] = [tweedie.tweedie(p=p, mu=mu, phi=phi).cdf(days) for days in range(1,31,1)]
        distros[mu][1:] = np.diff(distros[mu])
        distros[mu] = np.round(distros[mu] / np.sum(distros[mu]), 4)
    
    prob_array = np.zeros((pred_list.shape[0], 30))

    for row, mu in enumerate(pred_list):
        prob_array[row, :] = distros[mu]#.cumsum()
        #prob_array[row, -1] = 1.

    return prob_array

class XGBoostV1(Model):
    model_name = 'xgboost_v1'
    
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        self.df_train = read_df(os.path.join(self.dataset_path, self.default_paths['train_data']))
        self.df_item = read_df(self.default_paths['item_data'])
        
        self.prepared_dataset = self.df_train.join(self.df_item, how='left', on='sku', rsuffix="_item")
        self.prepared_dataset = self.prepared_dataset.drop('sku_item', axis=1)
        
        self.prepared_dataset['date'] = pd.to_datetime(self.prepared_dataset['date'])
        #self.prepared_dataset['fold'] = self.prepared_dataset['date'].dt.month
        
        enc = OrdinalEncoder(CATEGORICAL_FEATURES)
        self.prepared_dataset = enc.fit_transform(self.prepared_dataset)
    
    def train(self, params=DEFAULT_PARAMS):
        self.model = XGBRegressor(n_estimators=1000, learning_rate=params[0],
                   max_depth=params[1],
                   subsample=params[2],
                   colsample_bytree=params[3],
                   tweedie_variance_power=params[4],
                   min_child_weight=params[5],
                   random_state=0, objective="reg:tweedie", 
                   base_score=1e-3,
                   tree_method='gpu_hist', gpu_id=3)
        self.model.fit(self.prepared_dataset[FEATURES], self.prepared_dataset['sold_quantity'])
    
    def predict(self, df_test):
        test = read_df(df_test).set_index('sku').squeeze()
        
        #test_data = self.prepared_dataset[self.prepared_dataset['date'] == "2021-03-31"]
        #test_data = test_data[test_data['sku'].isin(test.index)]
        test_data = self.prepared_dataset[self.prepared_dataset['sku'].isin(test.index)]
        test_data = test_data.groupby('sku').last().reset_index()
        assert np.all(test_data['sku'] == test.index)
        
        p = self.model.predict(test_data[FEATURES])
        
        spp = test_data[['sku']].copy()
        spp['p'] = p
        spp['stock'] = spp['sku'].map(test)
        spp['days_to_so'] = (spp['stock'] / spp['p']).fillna(30.).clip(1,30).astype(int)

        prob_array = pred_list_to_tweedie(spp['days_to_so'].values, phi=2., p=1.5)
        
        return np.array(prob_array)
    
