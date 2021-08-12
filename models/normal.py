import numpy as np
import json

from tqdm.auto import tqdm
from multiprocessing import Pool
from iteround import saferound
from scipy.stats import norm

from model import Model
from utils import read_df

df_train_v1 = None
general_mean = None
general_std = None

def normal_probs(data):
    i, row = data
    sku = row['sku']
    target_stock = row['target_stock']
    train_row = df_train_v1.loc[sku]
    sold_quantity_series = np.array(json.loads(train_row['sold_quantity_series']))

    sold_mean = sold_quantity_series.mean()
    sold_std  = sold_quantity_series.std()

    if sold_mean == 0:
        sold_mean = general_mean
        sold_std = general_std
    if sold_std == 0:
        sold_std = general_std

    days_stockout = target_stock/sold_mean
    std_days = (sold_std / sold_mean) * days_stockout

    dist_model = norm(days_stockout,std_days)

    probalities = np.zeros(30)
    for i in range(1, 31):
        probalities[i-1] = (dist_model.cdf(i+1) - dist_model.cdf(i))

    if probalities.sum() == 0:
        probalities = np.ones(30) / 30

    probalities = probalities/probalities.sum()
    probalities = saferound(probalities, places=4)
    return (sku, probalities)

class Normal(Model):
    model_name = 'Normal'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        self.df_train_processed = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_processed']))
        self.prepared_dataset = self.df_train_processed
        
    def predict(self, df_test):
        global df_train_v1, general_mean, general_std
        df_train_v1 = self.df_train_processed.set_index('sku')
        df_test = read_df(df_test)
        
        general_mean = df_train_v1['sold_quantity_mean'].mean()
        general_std = df_train_v1['sold_quantity_std'].mean()

        predictions = []
        skus = []
        with Pool(100) as p:
            for data in tqdm(p.imap(normal_probs, df_test.iterrows()), total=len(df_test)):
                sku, probabilities = data
                skus.append(sku)
                predictions.append(probabilities)

        skus = np.array(skus)
        comparison = skus == df_test['sku'].to_numpy()
        assert comparison.all()
        
        return np.array(predictions)
    
