# Author: Bryan Lee
# Date: 2024-12-14

import pandas as pd
import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.train_test_split import run_TrainTestSplit

# Creating testing data
os.makedirs('test/temp/data/processed/', exist_ok=True)

test_clean_data_path = 'test/temp/data/processed/cleaned_data.csv'
test_output_path = 'test/temp/output/'

data = pd.DataFrame({
    'Col1': [1, 2, 3, 4, 2],
    'Col2': [1, 3, 2, 1, 3],
    'quality': [1, 0, 1, 0, 1]
})
data.to_csv(test_clean_data_path, index=False)

def test_train_test_split():
    # Run the train-test split function
    run_TrainTestSplit(test_clean_data_path, test_output_path)
    X_train = pd.read_csv(f"{test_output_path}X_train.csv")
    y_train = pd.read_csv(f"{test_output_path}y_train.csv")
    X_test = pd.read_csv(f"{test_output_path}X_test.csv")
    y_test = pd.read_csv(f"{test_output_path}y_test.csv")

    # test if the return values are DataFrames
    assert isinstance(
        X_train, pd.DataFrame), f"X_train is not a DataFrame, instead received: {type(X_train)}"
    assert isinstance(
        X_test, pd.DataFrame), f"X_test is not a DataFrame, instead received: {type(X_test)}"
    assert isinstance(
        y_train, (pd.Series, pd.core.frame.DataFrame)), f"y_train is not a Series, instead received: {type(y_train)}"
    assert isinstance(
        y_test, (pd.Series, pd.core.frame.DataFrame)), f"y_test is not a Series, instead received: {type(y_test)}"

    # test to check if X_train, y_train, X_test, y_test have the same number of observations
    assert X_train.shape[0] == y_train.shape[0], f"X_train and y_train row length not the same"
    assert X_test.shape[0] == y_test.shape[0], f"X_test and y_test row length not the same"

    # test to check if the df has a column named 'quality' to create target column
    assert y_test.columns[0]== 'quality', f"Test target column name is not 'quality'"

    # remove temporary files after test
    os.remove(f"{test_output_path}X_train.csv")
    os.remove(f"{test_output_path}y_train.csv")
    os.remove(f"{test_output_path}X_test.csv")
    os.remove(f"{test_output_path}y_test.csv")
    os.remove(f"{test_clean_data_path}")
    os.rmdir("test/temp/data/processed")
    os.rmdir("test/temp/data")
    os.rmdir(test_output_path)
