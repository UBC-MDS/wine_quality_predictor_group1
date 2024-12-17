# Author: Yixuan Gao
# Date: 2024-12-07

import click
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from clean import load_data, save_overview, handle_missing_values, handle_duplicates, save_cleaned_data


@click.command()
@click.option('--input_path', type=str, required=True, help="Path to the raw data file")
@click.option('--output_path', type=str, required=True, help="Path to save the cleaned data file")
@click.option('--log_path', type=str, required=True, help="Path to directory where logs will be saved")
def main(input_path, output_path, log_path):
    """
    Cleans data from a local relative path, saves the cleaned output, and logs details.

    Parameters:
    ----------
    input_path : str
        Relative path to the raw data file.
    output_path : str
        Relative path to save the cleaned data file.
    log_path : str
        Directory path to save the logs as CSV files.

    Returns:
    -------
    None
    """
    try:
        # Load the dataset
        df = load_data(input_path)
        
        # Save dataset overview
        save_overview(df, log_path)

        # Handle missing values
        handle_missing_values(df, log_path)

        # Handle duplicates
        df = handle_duplicates(df, log_path)

        # Save cleaned data
        save_cleaned_data(df, output_path)
        print(f"Cleaned data saved to {output_path}.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

