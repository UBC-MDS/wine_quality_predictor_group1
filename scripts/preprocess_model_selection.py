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

# Creating function to return accuracy of from each trained model
def cross_val_scores(model, X_train, y_train):
    """
    Returns mean accuracy from 5-fold cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        values for features from training data
    y_train : numpy array or pandas Series
        values for target from training data

    Returns
    ----------
        pandas Series with all mean scores from cross_validation
    """

    scores = cross_validate(model, 
                            X_train, 
                            y_train, 
                            cv = 5, 
                            return_train_score = True
                            )
    
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    result_scores =[]
    for i in range(len(scores)):
        result_scores.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))


    return pd.Series(data=result_scores, index=mean_scores.index)


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
    
        results[model_key] = cross_val_scores(model_pipeline,
                                                       X_train,
                                                       y_train)
    
    results_df = pd.DataFrame(results).T

    # Save results_df to .csv file to scores_path
    results_df.to_csv(os.path.join(scores_path, "inital_model_scores.csv"), index=True)
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
    
    

    
            