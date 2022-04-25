company_name=amc

python3 app.py \
    --training=./data/train_data/train_$company_name.csv \
    --testing=./data/train_data/test_$company_name.csv \
    --output=./data/output_data/output_$company_name.csv


python3 ./modules/profit_calculator.py \
    ./data/train_data/test_$company_name.csv \
    ./data/output_data/output_$company_name.csv