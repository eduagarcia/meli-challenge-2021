import json
import os

import numpy as np
import pandas as pd
from fastparquet import write
    
def read_df(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, list):
        return pd.DataFrame(list)
    elif isinstance(data, str):
        extension = os.path.splitext(data)[-1]
        if extension == '.csv':
            return pd.read_csv(data)
        elif extension == '.parquet':
            return pd.read_parquet(data, engine='fastparquet')
        elif extension == '.json':
            return pd.read_json(data)
        elif extension == '.jl' or extension == '.jsonl':
            return pd.read_json(data, lines=True)
    raise Exception('invalid data')
    
def write_df(df, path, index=False, header=True):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    extension = os.path.splitext(path)[-1]
    
    if extension == '.csv':
        df.to_csv(path, index=index, header=header)
        return path
    elif extension == '.parquet':     
        write(path, df)
        return path
    
    raise Exception('invalid extension '+path)
    
def read_numpy(data):
    if isinstance(data, np.ndarray):
        return data.copy()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, str):
        extension = os.path.splitext(data)[-1]
        if extension == '.csv':
            return pd.read_csv(data, header=None).to_numpy()
        elif extension == '.parquet':
            return pd.read_parquet(data, engine='fastparquet', header=None).to_numpy()
        elif extension == '.npy':
            return np.load(data)
    raise Exception('invalid data')
    
def read_json(data):
    if isinstance(data, dict):
        return data.copy()
    elif isinstance(data, list):
        return data.copy()
    elif isinstance(data, np.ndarray):
        return data.copy()
    elif isinstance(data, str):
        extension = os.path.splitext(data)[-1]
        if extension == '.json':
            with open(data, 'r') as f:
                data_loaded = json.load(f)
            return data_loaded
        elif extension == '.jl' or extension == '.jsonl':
            data_loaded = []
            with open(data, 'r') as f:
                for line in f:
                    data_loaded.append(json.loads(line))
            return data_loaded
        elif extension == '.npy':
            return np.load(data)
    raise Exception('invalid data') 