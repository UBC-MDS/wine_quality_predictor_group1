# Adapted from https://github.com/ttimbers/breast_cancer_predictor_py
# Author: Yixuan Gao
# Date: 2024-12-07


import click
import requests
import os
import zipfile

def download_file(url, output_path):
    """
    Download a file from the given URL and save it to the specified local path.

    Parameters:
    ----------
    url : str
        The URL of the file to download.
    output_path : str
        The local path where the file will be saved.

    Returns:
    -------
    None
    """
    response = requests.get(url, stream=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch data from URL: {url}. Error: {e}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory for output path {output_path}. Error: {e}")

    try:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except IOError as e:
        raise IOError(f"Failed to write to file {output_path}. Error: {e}")

def extract_specific_file(zip_path, target_file, output_path):
    """
    Extract a specific file from a ZIP archive.

    Parameters:
    ----------
    zip_path : str
        Path to the ZIP archive.
    target_file : str
        The name of the file to extract.
    output_path : str
        Path to save the extracted file.

    Returns:
    -------
    None
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if target_file not in zip_ref.namelist():
                raise ValueError(f"The target file {target_file} was not found in the ZIP archive.")

            # Extract only the specified file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with zip_ref.open(target_file) as src, open(output_path, 'wb') as dest:
                dest.write(src.read())
    except (zipfile.BadZipFile, KeyError) as e:
        raise ValueError(f"Failed to extract file {target_file} from {zip_path}. Error: {e}")
    except IOError as e:
        raise IOError(f"Failed to write extracted file to {output_path}. Error: {e}")

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
