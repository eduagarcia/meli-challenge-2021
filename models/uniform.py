import numpy as np

from model import Model
from utils import read_df

class Uniform(Model):
    model_name = 'uniform'
    
    def __init__(self, df_train, df_train_processed):
        Model.__init__(self, self.model_name, df_train, df_train_processed)
        
    def predict(self, df_test):
        df_test = read_df(df_test)
        
        predictions=np.empty((len(df_test), 30))
        predictions.fill(0.0333)
        predictions[:, -1] = 0.0343
        
        return predictions
    
