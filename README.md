# MeLi Data Challenge 2021

### Private Leaderboard Top 3 - 3.72303

The solution consists in extracting features from the time-series and using a XGBoost model to predict the probability distribution of the 30 days for each SKU.

The final submission as an ensemble (weighted mean) of 5 models with slight different input features.

### How to train the models, step by step:

```bash
#Install requirements
pip install -r requirements.txt

#Donwload the dataset into the ./dataset folder
./download_dataset.sh

#Prepare the training data files and extract features from each time series:
./prepare_dataset.sh ./dataset

#Select a model from the ./models folder to train and generate the submission file.
# e.g.:
python run.py xgboost_features2_v4_5_normalize

#The submission will be generated on the folder ./submissions
```

MercadoLibre Data Challenge 2021
