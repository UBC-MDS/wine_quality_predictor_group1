import numpy as np
import pandas as pd
import pickle
import click
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("best_model_path", type=str)
@click.argument("x_train_path", type=click.Path(exists=True))
@click.argument("y_train_path", type=click.Path(exists=True))
@click.argument("x_test_path", type=click.Path(exists=True))
@click.argument("y_test_path", type=click.Path(exists=True))
def main(model_path, best_model_path, x_train_path, y_train_path, x_test_path, y_test_path):
    """
    Fine-tunes a pre-trained model and saves the best model.

    MODEL_PATH: Path to the pre-trained model file (.pkl).
    BEST_MODEL_PATH: Path to save the fine-tuned model (.pkl).
    X_TRAIN_PATH: Path to the training features (CSV).
    Y_TRAIN_PATH: Path to the training labels (CSV).
    X_TEST_PATH: Path to the testing features (CSV).
    Y_TEST_PATH: Path to the testing labels (CSV).
    """
    # Load the saved model pipeline
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Define hyperparameter search space
    param_dist = {
        'svc__C': loguniform(1e-3, 1e3),
        'svc__gamma': loguniform(1e-3, 1e3),
        'svc__decision_function_shape': ['ovr', 'ovo'],
        'svc__class_weight': [None, 'balanced']
    }

    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        loaded_model, param_dist, n_iter=50, cv=5, n_jobs=-1, random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Output best hyperparameters and best cross-validation score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print(f"Best hyperparameters: {best_params}")
    print(f"Best cross-validation score: {best_score}")

    # Save the best model pipeline
    with open(best_model_path, "wb") as f:
        pickle.dump(random_search.best_estimator_, f)

    print(f"Best model saved to {best_model_path}")

    # Using the model with the best hyperparameters on the testing set
    test_score = random_search.best_estimator_.score(X_test, y_test)
    print(f"Test score with best model: {test_score}")

if __name__ == "__main__":
    main()
