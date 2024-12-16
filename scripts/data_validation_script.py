# Author: Wangkai Zhu
# Date: 2024-12-07

import click
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from validation import validate_dataset

@click.command()
@click.argument("input_path", type=click.Path(exists=True), nargs=1)
def main(input_path):
    """
    Validates the input CSV file against the predefined schema.
    
    input_path: Path to the CSV file to validate.
    """
    validate_dataset(input_path)

if __name__ == "__main__":
    main()
