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

    # Check if the URL exists and is accessible
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data from URL: {url}. HTTP status code: {response.status_code}")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the file locally
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

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
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if target_file not in zip_ref.namelist():
            raise ValueError(f"The target file {target_file} was not found in the ZIP archive.")

        # Extract only the specified file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with zip_ref.open(target_file) as src, open(output_path, 'wb') as dest:
            dest.write(src.read())

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
    temp_zip_path = os.path.join(write_to, "temp.zip")
    target_file = "winequality-red.csv"

    try:
        download_file(url, temp_zip_path)
        print(f"File successfully downloaded from {url} to {temp_zip_path}")

        # Extract the specific file
        extract_specific_file(temp_zip_path, target_file, os.path.join(write_to, target_file))
        print(f"Extracted {target_file} to {write_to}")

        # Remove the temporary ZIP file
        os.remove(temp_zip_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()