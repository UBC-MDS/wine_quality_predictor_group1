# Author: Timothy Singh
# Date: 2024-12-07

import pandas as pd
import numpy as np
import os
import click
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from multiconfusion_matrix import save_confusion_matrix_multi
from summarize_conf_matrix import summarize_conf_matrix

@click.command()
@click.option("--tuned_model_path", type=str, help="Path to access tuned model.")
@click.option("--test_split_path", type=str, help="Path to access testing data.")
@click.option("--test_accuracy_path", type=str, help="Path to save the test accuracy score.")
@click.option("--figures_path", type=str, help="Path to save any figures from evaluation.")
def main(tuned_model_path, test_split_path, test_accuracy_path, figures_path):
    """
    Finds the accuracy of the model for predictions on the testing set.
    Creates and saves confusion matrices using the One vs Rest method of comparison 
    from the multiconfusion_matrix.py function.

    INPUT:
    tuned_model_path: Relative path to the tuned model after hyperparameter tuning.
    test_split_path: Relative path to testing split of the data set.
    test_accuracy_path: Relative path to save test accuracy.
    figures_path: Path to save any figures from evaluation.
    """
    # Retrieve tuned model
    with open(tuned_model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    # Retrieve testing set
    X_test = pd.read_csv(f"{test_split_path}X_test.csv")
    y_test = pd.read_csv(f"{test_split_path}y_test.csv")

    # Calculate Test Score    
    test_score = best_model.score(X_test, y_test)
    test_accuracy_df = pd.DataFrame({"accuracy": [test_score]})

    #Ensuring path exists for saving figures and tables:
    os.makedirs(test_accuracy_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Save test_accuracy_df to .csv file to scores_path
    test_accuracy_df.to_csv(os.path.join(test_accuracy_path, "test_accuracy.csv"), index=False)
    print(f"Saved test_accuracy.csv to {test_accuracy_path}")

    # Generate and save multi-class confusion matrices
    confusion_matrix = save_confusion_matrix_multi(best_model, X_test, y_test, figures_path)
 
    # Create a DataFrame to summarize confusion matrices
    conf_matrix_summary = summarize_conf_matrix(confusion_matrix, np.unique(y_test))
    conf_matrix_summary.to_csv(os.path.join(test_accuracy_path, "confusion_matrix_summary.csv"))
    print(f"Saved confusion_matrix_summary.csv to {test_accuracy_path}")
        
if __name__ ==  "__main__":
    main()
            