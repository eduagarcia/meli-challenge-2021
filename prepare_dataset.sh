#!/bin/bash

set -e

DATA_PATH=${1}
#python feature_extraction.py ${DATA_PATH}/train_data.parquet ./dataset/items_static_metadata_full.jl ${DATA_PATH}/train_data_features.parquet
#python create_training_data.py ${DATA_PATH}/train_data.parquet ${DATA_PATH}/
#python feature_extraction.py ${DATA_PATH}/train_data_x.parquet ./dataset/items_static_metadata_full.jl ${DATA_PATH}/train_data_x_features.parquet
#python feature_extraction.py ${DATA_PATH}/train_data_y.parquet ./dataset/items_static_metadata_full.jl ${DATA_PATH}/train_data_y_features.parquet
python create_last_29_predict_data.py ${DATA_PATH}/train_data.parquet ${DATA_PATH}/test_data.csv ${DATA_PATH}/test_fromtrain_data_last29.parquet
python feature_extraction.py ${DATA_PATH}/test_fromtrain_data_last29.parquet ./dataset/items_static_metadata_full.jl ${DATA_PATH}/test_fromtrain_data_last29_features.parquet