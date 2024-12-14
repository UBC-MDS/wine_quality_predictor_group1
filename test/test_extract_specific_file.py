import pytest
import os
import zipfile
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from extract_specific_file import extract_specific_file

@pytest.fixture
def create_zip_file():
    # Set up a temporary ZIP file for testing
    zip_filename = 'test.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.writestr('file1.txt', 'This is the content of file1.txt.')
        zipf.writestr('file2.txt', 'This is the content of file2.txt.')
    yield zip_filename
    os.remove(zip_filename)  # Clean up after the test

@pytest.fixture
def cleanup_output():
    # Cleanup the output directory before and after tests
    output_path = 'output'
    if os.path.exists(output_path):
        for root, dirs, files in os.walk(output_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
    yield output_path

# Test for successfully extracting a file
def test_extract_file_success(create_zip_file, cleanup_output):
    extract_specific_file(create_zip_file, 'file1.txt', 'output/file1.txt')
    assert os.path.exists('output/file1.txt')

    with open('output/file1.txt', 'r') as f:
        content = f.read()
    assert content == 'This is the content of file1.txt.'

# Test for file not found in the zip archive
def test_file_not_found_in_zip(create_zip_file, cleanup_output):
    with pytest.raises(ValueError):
        extract_specific_file(create_zip_file, 'non_existent.txt', 'output/non_existent.txt')