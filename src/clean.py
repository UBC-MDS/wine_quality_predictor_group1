import os
import pandas as pd

def load_data(input_path):
    """
    Load the dataset from a specified path.

    Parameters:
    ----------
    input_path : str
        Path to the raw data file.

    Returns:
    -------
    pd.DataFrame
        Loaded dataset.
    """
    try:
        return pd.read_csv(input_path, sep=';')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The input file at {input_path} was not found. Error: {e}")

def save_overview(df, log_path):
    """
    Save dataset overview (columns, non-null counts, and dtypes) to a CSV file.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataset.
    log_path : str
        Path to save the overview CSV.

    Returns:
    -------
    None
    """
    os.makedirs(log_path, exist_ok=True)
    overview_file = os.path.join(log_path, "dataset_overview.csv")
    df_info = pd.DataFrame({"Column": df.columns, "Non-Null Count": df.count(), "Dtype": df.dtypes})
    df_info.to_csv(overview_file, index=False)

def handle_missing_values(df, log_path):
    """
    Generate and save a report on missing values in the dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataset.
    log_path : str
        Path to save the missing values report.

    Returns:
    -------
    None
    """
    os.makedirs(log_path, exist_ok=True)
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]
    missing_file = os.path.join(log_path, "missing_values.csv")
    missing_values.to_csv(missing_file, index=False)

def handle_duplicates(df, log_path):
    """
    Identify and save duplicates in the dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataset.
    log_path : str
        Path to save the duplicates report.

    Returns:
    -------
    pd.DataFrame
        Dataset with duplicates removed.
    """
    os.makedirs(log_path, exist_ok=True)
    duplicates = df[df.duplicated()].reset_index(drop=True)
    duplicates_file = os.path.join(log_path, "duplicates.csv")
    duplicates.to_csv(duplicates_file, index=False)
    return df.drop_duplicates()

def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataset to a specified path.

    Parameters:
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    output_path : str
        Path to save the cleaned dataset.

    Returns:
    -------
    None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
