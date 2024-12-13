# Author: Yixuan Gao
# Date: 2024-12-07

import click
import pandas as pd
import os

@click.command()
@click.option('--input_path', type=str, required=True, help="Path to the raw data file")
@click.option('--output_path', type=str, required=True, help="Path to save the cleaned data file")
@click.option('--log_path', type=str, required=True, help="Path to directory where logs will be saved")
def main(input_path, output_path, log_path):
    """
    Cleans data from a local relative path, saves the cleaned output, and logs details.
    
    INPUT:
    input_path: Relative path to the raw data file
    output_path: Relative path to save the cleaned data file
    log_path: Directory path to save the logs as CSV files
    """
    try:
        # Load the dataset
        try:
            df = pd.read_csv(input_path, sep=';')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The input file at {input_path} was not found. Error: {e}")

        # Prepare the log directory
        os.makedirs(log_path, exist_ok=True)

        # Save dataset overview
        overview_file = os.path.join(log_path, "dataset_overview.csv")
        df_info = pd.DataFrame({"Column": df.columns, "Non-Null Count": df.count(), "Dtype": df.dtypes})
        df_info.to_csv(overview_file, index=False)
        print(f"Dataset overview saved to {overview_file}.")

        # Handle missing values
        try:
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ["Column", "Missing Values"]
            missing_file = os.path.join(log_path, "missing_values.csv")
            missing_values.to_csv(missing_file, index=False)
            print(f"Missing values report saved to {missing_file}.")
        except IOError as e:
            raise IOError(f"Failed to save missing values report to {missing_file}. Error: {e}")

        # Remove duplicates
        duplicates = df[df.duplicated()].reset_index(drop=True)
        duplicates_file = os.path.join(log_path, "duplicates.csv")
        if not duplicates.empty:
            duplicates.to_csv(duplicates_file, index=False)
        else:
            pd.DataFrame().to_csv(duplicates_file, index=False)  # Save empty DataFrame if no duplicates
        print(f"Duplicates report saved to {duplicates_file}.")

        # Drop duplicates
        try:
            df = df.drop_duplicates()
        except Exception as e:
            raise ValueError(f"Failed to drop duplicates. Error: {e}")

        # Save the cleaned data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
