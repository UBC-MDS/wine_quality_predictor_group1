import os
import requests
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