import pytest
import os
import requests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from download_file import download_file

def test_download_file_success(tmp_path):
    url = "https://file-examples.com/wp-content/storage/2017/02/zip_2MB.zip"
    output_path = tmp_path / "zip_2MB.zip"

    # Simulate downloading a small file
    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            pytest.skip("Skipping test because URL is not reachable.")

    download_file(url, str(output_path))

    assert output_path.exists(), "File was not downloaded."
    assert output_path.stat().st_size > 0, "Downloaded file is empty."

def test_download_file_invalid_url(tmp_path):
    url = "https://example.com/nonexistentfile.zip"
    output_path = tmp_path / "zip_2MB.zip"

    with pytest.raises(ValueError, match="Failed to fetch data from URL"):
        download_file(url, str(output_path))