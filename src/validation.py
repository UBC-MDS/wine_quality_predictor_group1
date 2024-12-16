# Author: Wangkai Zhu
# Date: 2024-12-15

import pandera as pa
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation

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

def validate_dataset(input_path):
    """
    Validates the input CSV file against the predefined schema and performs deep checks.
    
    Parameters:
        input_path (str): Path to the CSV file to validate.
    """
    try:
        df = pd.read_csv(input_path)
        schema.validate(df)
        print("Dataset validation passed successfully.")

        # Incorporate deep check for feature-label correlation
        wine_ds = Dataset(df, label="quality", cat_features=[])
        check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
        check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=wine_ds)

        if not check_feat_lab_corr_result.passed_conditions():
            raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")

        print("Deepchecks validation passed successfully.")
    except pa.errors.SchemaError as e:
        print("Schema validation failed:")
        print(e)
        raise
    except FileNotFoundError:
        print("Dataset file not found. Please check the file path.")
        raise FileNotFoundError("Dataset file not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise