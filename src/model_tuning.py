import pickle
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

def fine_tune_model(
    model_path, 
    best_model_path, 
    x_train_path, 
    y_train_path, 
    x_test_path, 
    y_test_path, 
    params_output_path
):
    """
    Fine-tunes a pre-trained model and saves the best model and parameters.

    Parameters:
    - model_path: Path to the pre-trained model file (.pkl).
    - best_model_path: Path to save the fine-tuned model (.pkl).
    - x_train_path: Path to the training features (CSV).
    - y_train_path: Path to the training labels (CSV).
    - x_test_path: Path to the testing features (CSV).
    - y_test_path: Path to the testing labels (CSV).
    - params_output_path: Path to save the best parameters (CSV).
    """
    # Load the saved model pipeline
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Load datasets
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

    print("Finished Random Search")

    # Save the best model pipeline
    with open(best_model_path, "wb") as f:
        pickle.dump(random_search.best_estimator_, f)

    print(f"Best model saved to {best_model_path}")

    # Save best parameters to a CSV file
    best_params_df = pd.DataFrame([best_params])
    best_params_df['best_score'] = best_score
    best_params_df.to_csv(params_output_path, index=False)

    print(f"Best parameters saved to {params_output_path}")
