import click
import pandas as pd
import os

@click.command()
@click.argument('input_path', type=str)
@click.argument('output_path', type=str)
def main(input_path, output_path):
    """
    Handles data copying or downloading from a local relative path.
    
    INPUT:
    input_path: Relative local path to the data file
    output_path: Relative path to save the data file
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File {input_path} does not exist.")

        # Read and save the file
        data = pd.read_csv(input_path, sep=';')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Data copied from {input_path} to {output_path}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()