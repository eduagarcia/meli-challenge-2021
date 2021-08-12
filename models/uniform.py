import numpy as np

from model import Model
from utils import read_df

class Uniform(Model):
    model_name = 'uniform'
    
    def __init__(self, dataset_path):
        Model.__init__(self, self.model_name, dataset_path)
        
    def prepare_data(self):
        self.df_train_processed = read_df(os.path.join(self.dataset_path, self.default_paths['train_data_processed']))
        self.prepared_dataset = self.df_train_processed
        
    def predict(self, df_test):
        df_test = read_df(df_test)
        
        predictions=np.empty((len(df_test), 30))
        predictions.fill(0.0333)
        predictions[:, -1] = 0.0343
        
        return predictions
    
