# Author: Yixuan Gao
# Date: 2024-12-07

import click
import pandas as pd
import os

@click.command()
@click.argument('input_path', type=str)
@click.argument('output_path', type=str)
def main(input_path, output_path):
    """
    Cleans data from a local relative path and saves the cleaned output.
    
    INPUT:
    input_path: Relative path to the raw data file
    output_path: Relative path to save the cleaned data file
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_path, sep=';')
        print("Initial data loaded. Overview:")
        print(df.info())

        # Handle missing values
        missing_values = df.isnull().sum()
        print(f"Missing values:\n{missing_values}")

        # Remove duplicates
        duplicates = df[df.duplicated()]
        print(f"Number of duplicates: {len(duplicates)}")
        df = df.drop_duplicates()

        # Save the cleaned data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()