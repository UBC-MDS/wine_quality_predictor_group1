import pytest
import pandas as pd
import pandera as pa
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from validation import validate_dataset

# Define a helper function to create temporary CSV files
def create_temp_csv(data, file_name):
    df = pd.DataFrame(data)
    file_path = os.path.join(os.getcwd(), file_name)
    df.to_csv(file_path, index=False)
    return file_path

def test_valid_dataset():
    """
    Test that the function passes with a valid dataset.
    """
    data = {
        "fixed acidity": [7.4, 7.8],
        "volatile acidity": [0.7, 0.88],
        "citric acid": [0.0, 0.0],
        "residual sugar": [1.9, 2.6],
        "chlorides": [0.076, 0.098],
        "free sulfur dioxide": [11.0, 25.0],
        "total sulfur dioxide": [34.0, 67.0],
        "density": [0.9978, 0.9968],
        "pH": [3.51, 3.2],
        "sulphates": [0.56, 0.68],
        "alcohol": [9.4, 9.8],
        "quality": [5, 6],
    }
    file_path = create_temp_csv(data, "valid_dataset.csv")
    try:
        validate_dataset(file_path)
    finally:
        os.remove(file_path)

def test_schema_validation_failure():
    """
    Test that the function fails when the dataset violates the schema.
    """
    data = {
        "fixed acidity": [7.4, -7.8], 
        "volatile acidity": [0.7, 0.88],
        "citric acid": [0.0, 0.0],
        "residual sugar": [1.9, 2.6],
        "chlorides": [0.076, 0.098],
        "free sulfur dioxide": [11.0, 25.0],
        "total sulfur dioxide": [34.0, 67.0],
        "density": [0.9978, 0.9968],
        "pH": [3.51, 3.2],
        "sulphates": [0.56, 0.68],
        "alcohol": [9.4, 9.8],
        "quality": [5, 5],
    }
    file_path = create_temp_csv(data, "invalid_dataset.csv")
    with pytest.raises(pa.errors.SchemaError, match="Column 'fixed acidity' failed series or dataframe validator"):
        validate_dataset(file_path)
    os.remove(file_path)

def test_file_not_found():
    """
    Test that the function raises a FileNotFoundError for a non-existent file.
    """
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        validate_dataset("non_existent_file.csv")
        