# Author: Timothy Singh
# Date: 2024-12-14


import pytest
import os
import sys
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

# Import the get_save_confusion_matrix_multi function from multiconfusion_matrix
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.multiconfusion_matrix import save_confusion_matrix_multi

# Set up testing data
test_model = DummyClassifier()
test_X_train = pd.DataFrame({
    "A": np.random.randint(-10, 10, size=30),
    "B": np.random.randint(-20, 20, size=30),
    "C": np.random.randint(-5, 5, size=30),
})
test_y_train = pd.Series(["class_0"] * 10 + ["class_1"] * 10 + ["class_2"] * 10)
test_model.fit(test_X_train, test_y_train)

test_X_test = pd.DataFrame({
    "A": np.random.randint(-10, 10, size=30),
    "B": np.random.randint(-20, 20, size=30),
    "C": np.random.randint(-5, 5, size=30),
})
test_y_test = pd.Series(["class_0"] * 10 + ["class_1"] * 10 + ["class_2"] * 10)
test_save_path = "test/temp/"

test_X_test_empty = pd.DataFrame()
test_y_test_empty = pd.DataFrame()
test_save_path_empty = ""
test_save_path_numeric = 522
test_save_path_bool = False
test_X_test_numeric = 522
test_y_test_bool = True


# Testing that the return type is np.ndarray and saves images correctly
def test_save_confusion_matrix_multi_outputs_correct():
    result = save_confusion_matrix_multi(test_model, test_X_test, test_y_test, test_save_path)
    assert isinstance(result, np.ndarray), "save_confusion_matrix_multi should return a np.ndarray"
    assert os.path.exists(f'{test_save_path}confusion_matrix_class_class_0.png')
    assert os.path.exists(f'{test_save_path}confusion_matrix_class_class_1.png')
    assert os.path.exists(f'{test_save_path}confusion_matrix_class_class_2.png')

    #Clean files after testing
    os.remove(f"{test_save_path}confusion_matrix_class_class_0.png")
    os.remove(f"{test_save_path}confusion_matrix_class_class_1.png")
    os.remove(f"{test_save_path}confusion_matrix_class_class_2.png")

# Test for correct error handling for empty X_test or y_test or save_path
def test_save_confusion_matrix_empty_datasets_error():
    with pytest.raises(ValueError):
        save_confusion_matrix_multi(test_model, test_X_test_empty, test_y_test_empty, test_save_path)
        save_confusion_matrix_multi(test_model, test_X_test, test_y_test, test_save_path_empty)
        save_confusion_matrix_multi(test_model, test_X_test_empty, test_y_test_empty, test_save_path_empty)

# Test for correct error handling for wrong data types X_test or y_test or save_path.
def test_cross_val_scores_wrong_type_datasets_error():
    with pytest.raises(TypeError):
        save_confusion_matrix_multi(test_model, test_X_test, test_y_test, test_save_path_numeric)
        save_confusion_matrix_multi(test_model, test_X_test_numeric, test_y_test_bool, test_save_path)
        save_confusion_matrix_multi(test_model, test_X_test_numeric, test_y_test_bool, test_save_path_bool)

