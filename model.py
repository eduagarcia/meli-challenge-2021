from datetime import datetime
from utils import read_df

from evaluate import evaluate_rps

class Model():
    
    def __init__(self, name, df_train, df_train_processed):
        self.name = name
        self.df_train = read_df(df_train)
        self.df_train_processed = read_df(df_train_processed)
        self.date = datetime.now()
        
    def prepare_data(self):
        self.prepared_dataset = self.df_train_processed
        
    def train(self):
        pass
    
    def predict(self, df_test):
        return df_test
    
    def evaluate(self, prediction, ground_truth):
        return evaluate_rps(prediction, ground_truth)