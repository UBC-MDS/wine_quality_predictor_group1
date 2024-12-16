# Author: Timothy Singh
# Date: 2024-12-14

import pandas as pd
import numpy as np

def summarize_conf_matrix(confusion_matrix, labels):
    """
    Given a multi-class confusion matrix, creates a dataframe that
    shows the number of true positives, true negatives, false positives 
    and false negatives.

    Parameters
    ----------
    confusion_matrix : ndarray
        Multi-class confusion matrix
    labels : list
        List containing the possible targets in the classification

    Returns
    ----------
        Returns pandas dataframe with summarized data.
    """
    if not isinstance(confusion_matrix, np.ndarray):
        raise TypeError("confusion_matrix must be of type np.ndarray")
    
    if confusion_matrix.size == 0:
        raise ValueError("confusion_matrix must contain values.")
    
    if str(labels) == "":
        raise ValueError("labels cannot be blank.")

    columns = ["True Negative", "False Positive", "False Negative", "True Positive"]
    conf_matrix_summary_df = pd.DataFrame(confusion_matrix.reshape(len(labels), -1), index=labels, columns=columns).T
    return conf_matrix_summary_df
    

    