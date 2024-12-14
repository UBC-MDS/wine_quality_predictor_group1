import pytest
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from clean import load_data, save_overview, handle_missing_values, handle_duplicates, save_cleaned_data

@pytest.fixture
def sample_data():
    # Create a small sample dataset for testing
    data = {
        'Column1': [1, 2, 3, 4, None],
        'Column2': ['A', 'B', 'C', 'D', 'E'],
        'Column3': [None, None, 3, 4, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_directory():
    # Create a temporary directory for testing file operations
    temp_dir = 'test_temp_dir'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    yield temp_dir
    # Cleanup after test
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_dir)

# Test for loading data successfully
def test_load_data_success(temp_directory, sample_data):
    file_path = os.path.join(temp_directory, 'test_data.csv')
    sample_data.to_csv(file_path, index=False, sep=';')

    # Load the data using the load_data function
    df = load_data(file_path)

    # Test if the data is loaded correctly
    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_data.shape
    assert list(df.columns) == list(sample_data.columns)

# Test for handling FileNotFoundError
def test_load_data_file_not_found(temp_directory):
    invalid_file_path = os.path.join(temp_directory, 'invalid_data.csv')

    with pytest.raises(FileNotFoundError):
        load_data(invalid_file_path)

# Test for saving dataset overview
def test_save_overview(temp_directory, sample_data):
    save_overview(sample_data, temp_directory)

    # Check if the overview file is created
    overview_file = os.path.join(temp_directory, 'dataset_overview.csv')
    assert os.path.exists(overview_file)

    # Check if the file contains the correct overview
    df_overview = pd.read_csv(overview_file)
    assert set(df_overview['Column']) == set(sample_data.columns)

# Test for handling missing values
def test_handle_missing_values(temp_directory, sample_data):
    handle_missing_values(sample_data, temp_directory)

    # Check if the missing values report file is created
    missing_file = os.path.join(temp_directory, 'missing_values.csv')
    assert os.path.exists(missing_file)

    # Check if the report matches the expected missing values
    missing_values = pd.read_csv(missing_file)
    assert missing_values.shape == (len(sample_data.columns), 2)
    assert missing_values['Column'].iloc[0] == 'Column1'

# Test for handling duplicates
def test_handle_duplicates(temp_directory, sample_data):
    # Add a duplicate row to the sample data using pd.concat
    sample_data = pd.concat([sample_data, sample_data.iloc[0:1]], ignore_index=True)

    # Save the duplicates and clean the data
    cleaned_df = handle_duplicates(sample_data, temp_directory)

    # Check if the duplicates file is created
    duplicates_file = os.path.join(temp_directory, 'duplicates.csv')
    assert os.path.exists(duplicates_file)

    # Check that the duplicates file contains 1 duplicate row
    duplicates_df = pd.read_csv(duplicates_file)
    assert len(duplicates_df) == 1

    # Check if duplicates are removed from the cleaned data
    assert cleaned_df.shape[0] == sample_data.shape[0] - 1

# Test for saving cleaned data
def test_save_cleaned_data(temp_directory, sample_data):
    output_path = os.path.join(temp_directory, 'cleaned_data.csv')
    save_cleaned_data(sample_data, output_path)

    # Check if the cleaned data is saved
    assert os.path.exists(output_path)

    # Check if the content matches the input data
    saved_df = pd.read_csv(output_path)
    assert saved_df.shape == sample_data.shape
    assert list(saved_df.columns) == list(sample_data.columns)