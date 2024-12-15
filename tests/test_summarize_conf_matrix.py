# Author: Timothy Singh
# Date: 2024-12-14

import pytest
import os
import sys
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

# Import the get_save_confusion_matrix_multi function from multiconfusion_matrix
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.summarize_conf_matrix import summarize_conf_matrix

# Set up testing data
test_model = DummyClassifier()
test_X_train = pd.DataFrame({
    "A": np.random.randint(-10, 10, size=30),
    "B": np.random.randint(-20, 20, size=30),
    "C": np.random.randint(-5, 5, size=30),
})
test_y_train = ["class_0"] * 10 + ["class_1"] * 10 + ["class_2"] * 10
test_model.fit(test_X_train, test_y_train)

test_X_test = pd.DataFrame({
    "A": np.random.randint(-10, 10, size=30),
    "B": np.random.randint(-20, 20, size=30),
    "C": np.random.randint(-5, 5, size=30),
})
test_y_test = pd.Series(["class_0"] * 10 + ["class_1"] * 10 + ["class_2"] * 10) 
test_labels = np.unique(test_y_test)

test_y_pred = test_model.predict(test_X_test)
test_confusion_matrix = multilabel_confusion_matrix(test_y_test, test_y_pred, labels = test_labels)

test_confusion_matrix_str= "522"
test_confusion_matrix_numeric = 522
test_confusion_matrix_bool = True
test_confusion_matrix_empty = np.empty(0)

test_labels_empty = ""

# Testing that the return type of summarize_conf_matrix is pd.DataFrame
def test_summarize_conf_matrix_output_correct():
    result = summarize_conf_matrix(test_confusion_matrix, test_labels)
    assert isinstance(result, pd.DataFrame), "summarize_conf_matrix should return a pd.DataFrame"

# Testing that the returned DataFrame has the correct column names
def test_summarize_conf_matrix_output_correct_metric_names():
    result = summarize_conf_matrix(test_confusion_matrix, test_labels)
    expected_cols = ["True Negative", "False Positive", "False Negative", "True Positive"]
    assert list(result.T.columns) == expected_cols, "summarize_conf_matrix should return a pd.DataFrame containing: True Negative, False Positive, False Negative, True Positive."

# Test for correct error handling for wrong type of confusion_matrix variable
def test_summarize_conf_matrix_wrong_type_input():
    with pytest.raises(TypeError):
        summarize_conf_matrix(test_confusion_matrix_str, test_labels)
        summarize_conf_matrix(test_confusion_matrix_numeric, test_labels)
        summarize_conf_matrix(test_confusion_matrix_bool, test_labels)

# Test for correct error handling for empty confusion_matrix
def test_summarize_conf_matrix_empty_input():
    with pytest.raises(ValueError):
        summarize_conf_matrix(test_confusion_matrix_empty, test_labels)

# Test for correct error handling for empty labels
def test_summarize_conf_matrix_empty_labels_input():
    with pytest.raises(ValueError):
        summarize_conf_matrix(test_confusion_matrix, test_labels_empty)
