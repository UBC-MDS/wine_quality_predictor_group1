# Author: Timothy Singh
# Date: 2024-12-14

import pandas as pd
from sklearn.model_selection import cross_validate

def get_cross_val_scores(model, X_train, y_train):
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
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train should be of type pd.Dataframe. Got {type(X_train)}")
    
    if not (isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) or isinstance(y_train, pd.core.frame.DataFrame)):
        raise TypeError(f"y_train should be of type pd.Dataframe or pd.Series. Got {type(y_train)}")

    if X_train.shape[0] == 0:
        raise ValueError("The X_train DataFrame is empty.")
    
    if y_train.shape[0] == 0:
        raise ValueError("The y_train DataFrame is empty.")

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
