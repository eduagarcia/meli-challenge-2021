curl -o dataset/test_data.csv https://meli-data-challenge.s3.amazonaws.com/2021/test_data.csv
curl -o dataset/train_data.parquet https://meli-data-challenge.s3.amazonaws.com/2021/train_data.parquet
curl -o dataset/items_static_metadata_full.jl https://meli-data-challenge.s3.amazonaws.com/2021/items_static_metadata_full.jl
curl -o dataset/sample_submission.csv.gz https://meli-data-challenge.s3.amazonaws.com/2021/sample_submission.csv.gz
gunzip -kv dataset/sample_submission.csv.gz