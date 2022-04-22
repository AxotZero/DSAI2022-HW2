company_name=amc

python3 app.py --training=./data/train_$company_name.csv --testing=./data/test_$company_name.csv
# python3 app.py --training=./data/tesla_train.csv --testing=./data/tesla_test.csv

python3 ./modules/profit_calculator.py ./data/test_$company_name.csv output.csv