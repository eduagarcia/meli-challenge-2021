import numpy as np
import json
import os

from tqdm.auto import tqdm
from multiprocessing import Pool
from iteround import saferound
import scipy.stats as st
import tweedie

from model import Model
from utils import read_df

general_mean = None
general_std = None

def tweedie_probs(data):
    #i, row = data
    train_row = data
    sku = int(train_row['sku'])
    target_stock = int(train_row['target_stock'])
    #train_row = df_train_v1.loc[sku]
    #sold_quantity_series = np.array(json.loads(train_row['sold_quantity_series']))

    sold_mean = float(train_row['sold_quantity_mean'])
    sold_std  = float(train_row['sold_quantity_std'])

    if sold_mean == 0:
        sold_mean = general_mean
        sold_std = general_std
    if sold_std == 0:
        sold_std = general_std

    days_stockout = target_stock/sold_mean
    std_days = (sold_std / sold_mean) * days_stockout
    
    dist_model = tweedie.tweedie(p=1.5, mu=days_stockout, phi=2.)
    
    #probalities = [dist_model.cdf(days) for days in range(1,31,1)]
    #probalities[1:] = np.diff(probalities)
    #probalities = np.array(probalities)
    
    probalities = np.zeros(30)
    for i in range(1, 31):
        probalities[i-1] = (dist_model.cdf(i+1) - dist_model.cdf(i))
    
    if probalities.sum() == 0:
        probalities = np.ones(30) / 30

    probalities = (probalities/probalities.sum()).round(4)
    
    #probalities = saferound(probalities, places=4)
    return (sku, probalities)

class Tweedie(Model):
    model_name = 'tweedie'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        self.df_train_processed = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_processed']))
        self.prepared_dataset = self.df_train_processed.set_index('sku')
        
    def predict(self, df_test):
        global general_mean, general_std
        df_train_v1 = self.prepared_dataset
        df_test = read_df(df_test)
        
        general_mean = df_train_v1['sold_quantity_mean'].mean()
        general_std = df_train_v1['sold_quantity_std'].mean()
        
        df_train_v1 = df_train_v1.loc[df_test['sku']]
        df_train_v1['target_stock'] = df_test['target_stock'].values
        df_train_v1['sku'] = df_test['sku'].values
        
        df_train_v1 = df_train_v1[['sku', 'target_stock', 'sold_quantity_mean', 'sold_quantity_std']]
                
        predictions = []
        skus = []
        with Pool(100) as p:
            for data in tqdm(p.imap(tweedie_probs, df_train_v1.to_dict(orient='records')), total=len(df_train_v1)):
                sku, probabilities = data
                skus.append(sku)
                predictions.append(probabilities)

        skus = np.array(skus)
        comparison = skus == df_test['sku'].to_numpy()
        assert comparison.all()
        
        return np.array(predictions)
    
