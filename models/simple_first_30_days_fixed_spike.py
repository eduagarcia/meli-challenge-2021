import numpy as np
import json

from tqdm.auto import tqdm
from multiprocessing import Pool

from model import Model
from utils import read_df

df_train_v1 = None

def simple_first_30_days_fixed_spike(data):
    #row = df_test.iloc[0]
    i, row = data
    sku = row['sku']
    target_stock = row['target_stock']
    train_row = df_train_v1.loc[sku]
    sold_quantity_series = json.loads(train_row['sold_quantity_series'])

    original_len = len(sold_quantity_series)

    if original_len < 30:
        sold_quantity_series = np.pad(sold_quantity_series, (0, 30-original_len))

    sold_quantity_series = sold_quantity_series[:30]

    sold_quantity_cumsum = np.cumsum(sold_quantity_series)
    stock_percentage = sold_quantity_cumsum/target_stock
    stock_percentage = np.clip(stock_percentage,0,1)

    index_max = np.argmax(stock_percentage == stock_percentage.max())
    probalities = np.eye(30)[index_max]
    return (sku, probalities)

class SimpleFirst30DaysFixedSpike(Model):
    model_name = 'simple_first_30_days_fixed_spike'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        self.df_train_processed = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_processed']))
        self.prepared_dataset = self.df_train_processed
        
    def predict(self, df_test):
        global df_train_v1
        df_train_v1 = self.df_train_processed.set_index('sku')
        df_test = read_df(df_test)

        predictions = []
        skus = []
        with Pool(100) as p:
            for data in tqdm(p.imap(simple_first_30_days_fixed_spike, df_test.iterrows()), total=len(df_test)):
                sku, probabilities = data
                skus.append(sku)
                predictions.append(probabilities)

        skus = np.array(skus)
        comparison = skus == df_test['sku'].to_numpy()
        assert comparison.all()
        
        return np.array(predictions)
    
