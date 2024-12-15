# Author: Bryan Lee
# Date: 2024-12-14

import pandas as pd
import numpy as np
import pytest
import os
import zipfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train_test_split import run_TrainTestSplit
from eda_charts import run_eda_charts

def test_train_test_split(clean_data_path, train_test_path)

    # test if dataframe pull is dataframe object
    df = pd.read_csv(clean_data_path)
    assert isinstance(x, pd.DataFrame), f"Object returned is not dataframe, instead recieved: {type(df)}"
    
    X_train, X_test, y_train, y_test = run_TrainTestSplit(clean_data_path,train_test_path)

    # test to check if X_train, y_train, X_test, y_test have the same number of observations
    assert X_train.shape[0] == y_train.shape[0], f"X_train and y_train row length not the same"
    assert X_test.shape[0] == y_test.shape[0], f"X_train and y_train row length not the same"

    # testing if the exported csv file is functioning. 
    assert pd.read_csv(os.path.join(
        train_test_path, "X_train.csv")).shape[0] 
    == pd.read_csv(os.path.join(
        train_test_path, "y_train.csv")).shape[0], f"X_train and y_train row length not the same"
    
    assert pd.read_csv(os.path.join(
        train_test_path,"y_test.csv")).name == 'quality', f"test name is not quality"
        
