import pytest
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.eda_charts import run_eda_charts

# Sample file paths for testing
@pytest.fixture
def setup_directories():
    test_figures_path = "test/temp/test_figures/"
    test_tables_path = "test/temp/test_tables/"
    test_train_test_path = "test/temp/test_train_test/"
    
    # Create directories if they do not exist
    os.makedirs(test_figures_path, exist_ok=True)
    os.makedirs(test_tables_path, exist_ok=True)
    os.makedirs(test_train_test_path, exist_ok=True)

    # Creating mock training data files
    X_train = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [5, 4, 3, 2, 1]
    })
    y_train = pd.DataFrame({
        'Target': [1, 0, 1, 0, 1]
    })
    
    X_train.to_csv(os.path.join(test_train_test_path, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(test_train_test_path, 'y_train.csv'), index=False)
    
    return test_figures_path, test_tables_path, test_train_test_path

# Test case for checking directory creation
def test_directory_creation(setup_directories):
    test_figures_path, test_tables_path, test_train_test_path = setup_directories

    # Run the EDA charts function
    run_eda_charts(test_figures_path, test_tables_path, test_train_test_path)
    
    # Check if directories are created
    assert os.path.exists(test_figures_path), "Figures directory was not created."
    assert os.path.exists(test_tables_path), "Tables directory was not created."

# Test case for checking file generation
def test_file_generation(setup_directories):
    test_figures_path, test_tables_path, test_train_test_path = setup_directories

    # Run the EDA charts function
    run_eda_charts(test_figures_path, test_tables_path, test_train_test_path)

    # Check if files are generated
    assert os.path.exists(os.path.join(test_figures_path, "target_distribution_plot.png")), "Target distribution plot was not saved."
    assert os.path.exists(os.path.join(test_figures_path, "correlation_heatmap.png")), "Correlation heatmap was not saved."
    assert os.path.exists(os.path.join(test_figures_path, "feature_distributions.png")), "Feature distributions plot was not saved."
    assert os.path.exists(os.path.join(test_figures_path, "feature_pairplots.png")), "Feature pairplot was not saved."
    assert os.path.exists(os.path.join(test_tables_path, "describe_table.csv")), "Describe table was not saved."

# Test case for checking content in the describe table (CSV)
def test_describe_table_content(setup_directories):
    test_figures_path, test_tables_path, test_train_test_path = setup_directories

    # Run the EDA charts function
    run_eda_charts(test_figures_path, test_tables_path, test_train_test_path)

    # Check if the describe table exists
    describe_table_path = os.path.join(test_tables_path, "describe_table.csv")
    assert os.path.exists(describe_table_path), "Describe table was not saved."

    # Load the describe table and check if it contains expected columns
    describe_df = pd.read_csv(describe_table_path)
    assert 'Feature1' in describe_df.columns, "Feature1 is missing in describe table."
    assert 'Feature2' in describe_df.columns, "Feature2 is missing in describe table."

# Test case for ensuring that no exceptions are raised
def test_no_exceptions(setup_directories):
    test_figures_path, test_tables_path, test_train_test_path = setup_directories

    try:
        # Run the EDA charts function and assert no exceptions are raised
        run_eda_charts(test_figures_path, test_tables_path, test_train_test_path)
    except Exception as e:
        assert False, f"An exception occurred during EDA charts generation: {e}"

# Cleanup after all tests
def test_cleanup_after_tests(setup_directories):
    test_figures_path, test_tables_path, test_train_test_path = setup_directories
    
    # remove temporary files after test
    os.remove(f"{test_train_test_path}X_train.csv")
    os.remove(f"{test_train_test_path}y_train.csv")
    os.remove(f"{test_figures_path}target_distribution_plot.png")
    os.remove(f"{test_figures_path}correlation_heatmap.png")
    os.remove(f"{test_figures_path}feature_distributions.png")
    os.remove(f"{test_figures_path}feature_pairplots.png")
    os.remove(f"{test_tables_path}describe_table.csv")
    os.rmdir(test_figures_path)
    os.rmdir(test_tables_path)
    os.rmdir(test_train_test_path)
    

