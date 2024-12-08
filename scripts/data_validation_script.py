# Author: Wangkai Zhu
# Date: 2024-12-07

import pandera as pa
import pandas as pd
import click

# Define the DataFrame schema
schema = pa.DataFrameSchema(
    {
        "fixed acidity": pa.Column(float, pa.Check(lambda s: (s > 0).all()), nullable=False),
        "volatile acidity": pa.Column(float, pa.Check(lambda s: (s > 0).all()), nullable=False),
        "citric acid": pa.Column(float, pa.Check(lambda s: (s >= 0).all()), nullable=False),
        "residual sugar": pa.Column(float, pa.Check(lambda s: (s >= 0).all()), nullable=False),
        "chlorides": pa.Column(float, pa.Check(lambda s: (s >= 0).all()), nullable=False),
        "free sulfur dioxide": pa.Column(float, pa.Check(lambda s: (s >= 0).all()), nullable=False),
        "total sulfur dioxide": pa.Column(float, pa.Check(lambda s: (s >= 0).all()), nullable=False),
        "density": pa.Column(float, pa.Check(lambda s: ((s >= 0.9) & (s <= 1.1)).all()), nullable=False),
        "pH": pa.Column(float, pa.Check(lambda s: ((s >= 0) & (s <= 14)).all()), nullable=False),
        "sulphates": pa.Column(float, pa.Check(lambda s: (s >= 0).all()), nullable=False),
        "alcohol": pa.Column(float, pa.Check(lambda s: ((s >= 5) & (s <= 20)).all()), nullable=False),
        "quality": pa.Column(int, pa.Check.isin(range(0, 11)), nullable=False),
    },
    checks=[
        # Check for duplicate rows at the DataFrame level
        pa.Check(lambda df: not df.duplicated().any(), error="Duplicate rows found."),
        # Check for empty rows (rows with all NaN values)
        pa.Check(lambda df: not (df.isna().all(axis=1)).any(), error="Empty rows found."),
    ]
)

@click.command()
@click.argument("input_path", type=click.Path(exists=True), nargs=1)
def main(input_path):
    """
    Validates the input CSV file against the predefined schema.
    
    input_path: Path to the CSV file to validate.
    """
    try:
        df = pd.read_csv(input_path)
        schema.validate(df)
        print("Dataset validation passed successfully.")
    # if validation throws an error
    except pa.errors.SchemaError as e:
        print("Schema validation failed:")
        print(e)
    # if file import throws an error
    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
