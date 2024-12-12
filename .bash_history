cd
python scripts/download_data.py --url=https://archive.ics.uci.edu/static/public/186/wine+quality.zip --write_to=data/raw/
python scripts/clean_data.py --input_path=data/raw/raw_data.csv --output_path=data/processed/cleaned_data.csv --log_path=results/tables/
python scripts/data_validation_script.py data/processed/cleaned_data.csv
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
