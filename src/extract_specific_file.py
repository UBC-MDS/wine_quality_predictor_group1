import os
import requests
import zipfile

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