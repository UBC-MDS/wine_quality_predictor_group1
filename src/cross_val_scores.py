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
