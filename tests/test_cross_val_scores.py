# Author: Timothy Singh
# Date: 2024-12-14

import pytest
import os
import sys
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

# Import the get_cross_val_scores function from cross_val_scores
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.cross_val_scores import get_cross_val_scores

# Set up testing data
test_model = DummyClassifier()
test_X_train = pd.DataFrame({
    "A": np.random.randint(-10, 10, size=30),
    "B": np.random.randint(-20, 20, size=30),
    "C": np.random.randint(-5, 5, size=30),
})
test_y_train = pd.Series(["class_0"] * 10 + ["class_1"] * 10 + ["class_2"] * 10)

test_X_train_empty = pd.DataFrame()
test_y_train_empty = pd.DataFrame()

test_X_train_numeric = 522
test_y_train_bool = True

# Testing that the return type is pd.Series.
def test_cross_val_scores_returns_series():
    result = get_cross_val_scores(test_model, test_X_train, test_y_train)
    assert isinstance(result, pd.Series), "cross_val_scores should return a pandas Series"


# Test for correct number of rows in the Series.
# This should always be 4: fit time, score time, test score, train score
def test_cross_val_scores_number_of_rows():
    assert get_cross_val_scores(test_model, test_X_train, test_y_train).shape[0] == 4


# Test for correct error handling for empty X_train or y_train.
def test_cross_val_scores_empty_datasets_error():
    with pytest.raises(ValueError):
        get_cross_val_scores(test_model, test_X_train_empty, test_y_train)
        get_cross_val_scores(test_model, test_X_train, test_y_train_empty)
        get_cross_val_scores(test_model, test_X_train_empty, test_y_train_empty)


# Test for correct error handling for wrong data types X_train or y_train.
def test_cross_val_scores_wrong_type_datasets_error():
    with pytest.raises(TypeError):
        get_cross_val_scores(test_model, test_X_train_numeric, test_y_train_bool)