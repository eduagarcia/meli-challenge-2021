import numpy as np
import json

from tqdm.auto import tqdm
from multiprocessing import Pool
from iteround import saferound

from model import Model
from utils import read_df

df_train_v1 = None

def shift(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def gaussian_kernel1d(sigma, length, order=0):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(np.floor(-length/2), np.ceil(length/2))
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x


def voted_shifted_padded_gaussian_probs(data):
    i, row = data
    sku = row['sku']
    target_stock = row['target_stock']
    train_row = df_train_v1.loc[sku]
    sold_quantity_series = json.loads(train_row['sold_quantity_series'])

    original_len = len(sold_quantity_series)

    if original_len < 59:
        sold_quantity_series = np.pad(sold_quantity_series, (0, 59-original_len))

    #sold_quantity_series = sold_quantity_series[:30]
    voted_probalities = np.zeros(30)
    for i in range(original_len):
        shifted_sum_sold_quantity_series = shift(sold_quantity_series, -i)[:30]

        sold_quantity_cumsum = np.cumsum(shifted_sum_sold_quantity_series)
        stock_percentage = sold_quantity_cumsum/target_stock
        stock_percentage_clipped = np.clip(stock_percentage,0,1) 

        index_max = np.argmax(stock_percentage_clipped == stock_percentage_clipped.max())
        shifted_probalities = np.eye(30)[index_max]
        shifted_probalities *= stock_percentage_clipped.max()
        #shifted_probalities *= stock_percentage[index_max]
        voted_probalities += shifted_probalities

    index_max = np.argmax(voted_probalities == voted_probalities.max())
    #probalities = np.eye(30)[index_max]

    if (voted_probalities == np.zeros(30)).all():
        voted_probalities[0] = 1

    gaussian_len = 30
    sigma = np.sqrt(voted_probalities.std())
    sigma = sigma if sigma > 0 else 1
    gaussian = gaussian_kernel1d(sigma, gaussian_len)
    gaussian_central_point = int(np.floor(gaussian_len/2))
    shift_amount = index_max-gaussian_central_point

    probalities = voted_probalities*shift(gaussian, shift_amount)

    probalities = probalities/probalities.sum()
    probalities = saferound(probalities, places=4)
    return (sku, probalities)

class VotedShiftedPaddedGaussianProbs(Model):
    model_name = 'voted_shifted_padded_gaussian_probs'
    
    def __init__(self, df_train, df_train_processed):
        Model.__init__(self, self.model_name, df_train, df_train_processed)
        
    def predict(self, df_test):
        global df_train_v1
        df_train_v1 = self.df_train_processed.set_index('sku')
        df_test = read_df(df_test)

        predictions = []
        skus = []
        with Pool(100) as p:
            for data in tqdm(p.imap(voted_shifted_padded_gaussian_probs, df_test.iterrows()), total=len(df_test)):
                sku, probabilities = data
                skus.append(sku)
                predictions.append(probabilities)

        skus = np.array(skus)
        comparison = skus == df_test['sku'].to_numpy()
        assert comparison.all()
        
        return np.array(predictions)
    
