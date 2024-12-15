# Author: Timothy Singh
# Date: 2024-12-07

import pandas as pd
import click
import pickle
import os
from sklearn.model_selection import cross_validate
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from cross_val_scores import get_cross_val_scores

@click.command()
@click.option("--train_data_path", type=str, help="Relative path to retrieve training data.")
@click.option("--scores_path", type=str, help="Relative path to save training and validation scores.")
@click.option("--preprocessor_path", type=str, help="Relative path to save the preprocessor as .pickle file.")
@click.option("--model_path", type=str, help="Relative path to save best performing model as .pickle file.")
def main(train_data_path, scores_path, preprocessor_path, model_path):
    """
    Creates preprocessor and pipelines, and evaluates the performance of different models on the training data. 
    Dumps the model with the best evaluation score as a .pickle file.
    
    INPUT:
    train_data_path: Relative path to retrieve training data.
    scores_path: Relative path to save training and validation scores.
    preprocessor_path: Relative path to save the preprocessor as .pickle file.
    model_path: Relative path to save best performing model as .picklefile.
    """

    # Ensuring file paths exists
    os.makedirs(scores_path, exist_ok=True)
    os.makedirs(preprocessor_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Loading training set
    X_train = pd.read_csv(f"{train_data_path}X_train.csv")
    y_train = pd.read_csv(f"{train_data_path}y_train.csv")
    
    # Creating Column Transformer
    numeric_features = list(X_train.columns)
    preprocessor = make_column_transformer((StandardScaler(), numeric_features))
    pickle.dump(preprocessor, open(f"{preprocessor_path}preprocessor.pickle", 'wb'))
    print(f"Successfully saved preprocessor to {preprocessor_path}.")

    # Creation of model dictionary
    models = {
        "dummy": DummyClassifier(),
        "decision tree": DecisionTreeClassifier(),
        "kNN": KNeighborsClassifier(),
        "RBF SVM": SVC(),
        "naive bayes": GaussianNB(),
        "log reg": LogisticRegression()
    }
    
    # Creation of results dictionary
    results= {}

    # Creation of model pipelines and cross-evaluation
    for model_key, model in models.items():
        model_pipeline = make_pipeline(
            preprocessor,
            model
        )
    
        results[model_key] = get_cross_val_scores(model_pipeline,
                                                       X_train,
                                                       y_train)
    
    results_df = pd.DataFrame(results).T

    # Save results_df to .csv file to scores_path
    results_df.to_csv(os.path.join(scores_path, "initial_model_scores.csv"), index=True)
    print(f"Successfully saved cross validation results to {scores_path}.")

    # Save model pipeline with best validation score to model_path
    model_name = results_df["test_score"].idxmax()
    model = make_pipeline(
            preprocessor,
            models[model_name]
        )
    pickle.dump(model, open(f"{model_path}base_model.pickle", "wb"))
    print(f"Successfully saved {model_name} model to {model_path}.")

if __name__ ==  "__main__":
    main()
    
    

    
            