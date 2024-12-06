import click
import pandas as pd

@click.command()
@click.option('--input_path', type=str, help="Path to input file (URL or local path).")
@click.option('--output_path', type=str, help="Path where the downloaded file will be saved.")
def download_data(input_path, output_path):
    """
    Downloads or reads data from the specified input_path and saves it to the output_path.
    """
    try:
        df = pd.read_csv(input_path, sep=';')
        df.to_csv(output_path, index=False)
        click.echo(f"Data successfully downloaded and saved to {output_path}")
    except Exception as e:
        click.echo(f"Error in downloading data: {e}")

if __name__ == '__main__':
    download_data()