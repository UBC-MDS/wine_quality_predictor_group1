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
@click.command("--input_path", type=str, help="Path to import data.")
def main(input_path):
    try:
        df = pd.read_csv('input_path')
        schema.validate(df)
        print("Dataset is valid.")
    except pa.errors.SchemaError as e:
        print("Schema validation failed:", e)
    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")

if __name__ == "__main__":
    main()
