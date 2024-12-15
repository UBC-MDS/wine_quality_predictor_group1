# Author: Bryan Lee
# Date: 2024-12-07

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Train-test split
def run_TrainTestSplit(clean_data_path, train_test_path):
    """
    Splits a dataset into training and testing subsets and saves them as CSV files.

    This function reads a cleaned dataset from the specified file path, separates the features 
    (X) and the target variable (y), performs a train-test split, and saves the resulting 
    subsets (X_train, y_train, X_test, y_test) into the specified directory. The function 
    handles potential errors during file reading and data splitting.

    Parameters:
    ----------
    clean_data_path : str
        The file path to the cleaned dataset in CSV format. The dataset is expected to have 
        a 'quality' column as the target variable and the rest of the columns as features.

    train_test_path : str
        The directory path where the train-test split CSV files (X_train, y_train, X_test, 
        y_test) will be saved. If the directory doesn't exist, it will be created.

    Returns:
    -------
    None
        This function does not return any value but will print success or error messages.

    Raises:
    ------
    Exception:
        If any error occurs during the loading of the CSV, train-test splitting, or file 
        saving, an exception message will be printed.
    """
    try:
        # Load data
        df = pd.read_csv(clean_data_path)
    except Exception as e:
        print(f'issue with reading csv: {e}') 

    try:
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        os.makedirs(train_test_path, exist_ok=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train.to_csv(os.path.join(train_test_path, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(train_test_path, "y_train.csv"), index=False)
        X_test.to_csv(os.path.join(train_test_path, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(train_test_path, "y_test.csv"), index=False)
        print('Train-test split functioning properly')
    except Exception as e:
        print(f"Unexpected error during train-test split: {e}")