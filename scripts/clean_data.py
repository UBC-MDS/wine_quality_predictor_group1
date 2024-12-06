import click
import pandas as pd

@click.command()
@click.option('--input_path', type=str, help="Path to input CSV file to be cleaned.")
@click.option('--output_path', type=str, help="Path where the cleaned file will be saved.")
def clean_data(input_path, output_path):
    """
    Cleans the dataset by removing duplicates and handling missing values.
    """
    try:
        df = pd.read_csv(input_path)
        click.echo("Initial data info:")
        click.echo(df.info())
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        click.echo(f"Removed {initial_count - final_count} duplicate rows.")

        # Save cleaned data
        df.to_csv(output_path, index=False)
        click.echo(f"Cleaned data saved to {output_path}")
    except Exception as e:
        click.echo(f"Error in cleaning data: {e}")

if __name__ == '__main__':
    clean_data()