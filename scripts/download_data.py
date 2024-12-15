# Adapted from https://github.com/ttimbers/breast_cancer_predictor_py
# Author: Yixuan Gao
# Date: 2024-12-07


import click
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from download_file import download_file
from extract_specific_file import extract_specific_file


@click.command()
@click.option("--url", type=str, help="URL to download data as zip file from the Internet.")
@click.option("--write_to", type=str, help="Path to save downloaded zip file.")
def main(url, write_to):
    """
    Downloads data zip data from the web to a local filepath and extracts it.

    Parameters:
    ----------
    url : str
        The URL of the zip file to download.
    write_to : str
        The directory where the contents of the zip file will be extracted.

    Returns:
    -------
    None
    """
    zip_path = os.path.join(write_to, "raw_data.zip")
    target_file = "winequality-red.csv"
    raw_data_file = "raw_data.csv"

    try:
        download_file(url, zip_path)
        print(f"File successfully downloaded from {url} to {zip_path}")

        # Extract the specific file
        extracted_file_path = os.path.join(write_to, target_file)
        extract_specific_file(zip_path, target_file, extracted_file_path)
        print(f"Extracted {target_file} to {write_to}")
        
        # Rename the extracted file
        renamed_file_path = os.path.join(write_to, raw_data_file)
        try:
            os.rename(extracted_file_path, renamed_file_path)
            print(f"Renamed {target_file} to {raw_data_file}")
        except OSError as e:
            raise OSError(f"Failed to rename file {extracted_file_path} to {renamed_file_path}. Error: {e}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()