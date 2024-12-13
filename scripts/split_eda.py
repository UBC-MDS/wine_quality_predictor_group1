# Author: Bryan Lee
# Date: 2024-12-07

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import click

from src.split_eda import run_TrainTestSplit, run_eda_charts

@click.command()
@click.option("--clean_data_path", type=str, help="Path to pull raw data for train_test_split.")
@click.option("--train_test_path", type=str, help="Path to store and access data splits.")
@click.option("--figures_path", type=str, help="Path to save figures generated.")
@click.option("--tables_path", type=str, help="Path to save any tables generated")
def main(clean_data_path, train_test_path, figures_path, tables_path):
    """
    The main function for reading CSV from path, performing train-test split to create our training and testing 
    and creating our EDA visualizations.
    """
    run_TrainTestSplit(clean_data_path, train_test_path)
    run_eda_charts(figures_path, tables_path, train_test_path)

if __name__ == '__main__':
    main()