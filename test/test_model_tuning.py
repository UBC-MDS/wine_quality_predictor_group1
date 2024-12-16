import pytest
import pandas as pd
import pickle
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from model_tuning import fine_tune_model


def create_mock_data(file_path, data):
    """Helper function to create CSV files for testing."""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def create_mock_model(file_path):
    """Helper function to create a mock model pipeline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ])
    with open(file_path, 'wb') as f:
        pickle.dump(pipeline, f)


@pytest.fixture
def setup_mock_files(tmpdir):
    """Fixture to set up mock data and model files."""
    model_path = tmpdir.join("mock_model.pkl")
    best_model_path = tmpdir.join("best_model.pkl")
    params_output_path = tmpdir.join("params_output.csv")
    x_train_path = tmpdir.join("x_train.csv")
    y_train_path = tmpdir.join("y_train.csv")
    x_test_path = tmpdir.join("x_test.csv")
    y_test_path = tmpdir.join("y_test.csv")

    # Create mock model
    create_mock_model(model_path)

    # Create mock training and testing datasets
    x_train = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    }
    y_train = {'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    x_test = {
        'feature1': [11, 12, 13, 14, 15], 
        'feature2': [15, 16, 17, 18, 19]
    }
    y_test = {'label': [0, 1, 0, 1, 0]}
    
    create_mock_data(x_train_path, x_train)
    create_mock_data(y_train_path, y_train)
    create_mock_data(x_test_path, x_test)
    create_mock_data(y_test_path, y_test)

    return {
        'model_path': str(model_path),
        'best_model_path': str(best_model_path),
        'params_output_path': str(params_output_path),
        'x_train_path': str(x_train_path),
        'y_train_path': str(y_train_path),
        'x_test_path': str(x_test_path),
        'y_test_path': str(y_test_path),
    }


def test_fine_tune_model_success(setup_mock_files):
    """
    Test that the function successfully fine-tunes a model and saves outputs.
    """
    paths = setup_mock_files

    # Call the function
    fine_tune_model(
        model_path=paths['model_path'],
        best_model_path=paths['best_model_path'],
        x_train_path=paths['x_train_path'],
        y_train_path=paths['y_train_path'],
        x_test_path=paths['x_test_path'],
        y_test_path=paths['y_test_path'],
        params_output_path=paths['params_output_path']
    )

    # Check that the best model and parameters are saved
    assert os.path.exists(paths['best_model_path']), "Best model file not found."
    assert os.path.exists(paths['params_output_path']), "Parameters output file not found."

    # Verify the content of the parameters file
    params_df = pd.read_csv(paths['params_output_path'])
    assert 'best_score' in params_df.columns, "Best score missing in parameters output."


def test_inconsistent_data_raises_error(setup_mock_files):
    """
    Test that the function raises an error when X_train and y_train have inconsistent lengths.
    """
    paths = setup_mock_files

    # Create inconsistent training data
    x_train = {'feature1': [1, 2, 3], 'feature2': [5, 6, 7]}  # 3 rows
    y_train = {'label': [0, 1, 0, 1]}  # 4 rows
    create_mock_data(paths['x_train_path'], x_train)
    create_mock_data(paths['y_train_path'], y_train)

    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples:"):
        fine_tune_model(
            model_path=paths['model_path'],
            best_model_path=paths['best_model_path'],
            x_train_path=paths['x_train_path'],
            y_train_path=paths['y_train_path'],
            x_test_path=paths['x_test_path'],
            y_test_path=paths['y_test_path'],
            params_output_path=paths['params_output_path']
        )


def test_invalid_model_path_raises_error(setup_mock_files):
    """
    Test that the function raises an error when the model file does not exist.
    """
    paths = setup_mock_files
    invalid_model_path = "non_existent_model.pkl"

    with pytest.raises(FileNotFoundError):
        fine_tune_model(
            model_path=invalid_model_path,
            best_model_path=paths['best_model_path'],
            x_train_path=paths['x_train_path'],
            y_train_path=paths['y_train_path'],
            x_test_path=paths['x_test_path'],
            y_test_path=paths['y_test_path'],
            params_output_path=paths['params_output_path']
        )
