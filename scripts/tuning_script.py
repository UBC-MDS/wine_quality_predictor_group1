# Author: Wangkai Zhu
# Date: 2024-12-07

import click
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_tuning import fine_tune_model

@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("best_model_path", type=str)
@click.argument("x_train_path", type=click.Path(exists=True))
@click.argument("y_train_path", type=click.Path(exists=True))
@click.argument("x_test_path", type=click.Path(exists=True))
@click.argument("y_test_path", type=click.Path(exists=True))
@click.argument("params_output_path", type=str)
def main(model_path, best_model_path, x_train_path, y_train_path, x_test_path, y_test_path, params_output_path):
    """
    Fine-tunes a pre-trained model and saves the best model.

    model_path: Path to the pre-trained model file (.pkl).
    best_model_path: Path to save the fine-tuned model (.pkl).
    x_train_path: Path to the training features (CSV).
    y_train_path: Path to the training labels (CSV).
    x_test_path: Path to the testing features (CSV).
    y_test_path: Path to the testing labels (CSV).
    params_output_path: Path to save the best parameters (CSV).
    """
    fine_tune_model(
        model_path, 
        best_model_path, 
        x_train_path, 
        y_train_path, 
        x_test_path, 
        y_test_path, 
        params_output_path
    )

if __name__ == "__main__":
    main()
