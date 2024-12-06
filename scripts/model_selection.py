import pandas as py
import click
import pickle

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
@click.option("--scores_path", type=str, help="Path to save training and validation scores.")
@click.option("--preprocessor_path", type=str, help="Path to save the preprocessor as .pkl file.")
@click.option("--model_path", type=str, help="Path to save best performing model as .pkl file.")
def main(scores_path, preprocessor_path, model_path):

    # Loading training set
    # X_train = ...
    # y_train = ...
    
    # Creating Column Transformer
    numeric_features = list(X_train.columns)
    preprocessor = make_column_transformer((StandardScaler(), numeric_features))
    pickle.dump(preprocessor, open(model_path + "preprocessor.pickle", 'wb')) 

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
    write_csv(results_df, scores_path, "inital_model_scores.csv")

    # Save model pipeline with best validation score to model_path
    model_name = results_df["test_score"].idxmax()
    model = make_pipeline(
            preprocessor,
            models[model_name]
        )
    pickle.dump(model, open(model_path + "base_model.pickle", "wb"))

if __name__ ==  "__main__":
    main()
    
    

    
            