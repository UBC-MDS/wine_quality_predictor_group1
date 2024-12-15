# Author: Timothy Singh
# Date: 2024-12-14

import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Multi-class Confusion Matrix
def save_confusion_matrix_multi(model, X_test, y_test, save_path):
    """
    Given a model and X_test and y_test from the test split, multi-class confusion matrices 
    are created, using One vs Rest, where the correct predictions of each class are compared to all
    other classes.

    Parameters
    ----------
    model :
        scikit-learn model
    X_test : numpy array or pandas DataFrame
        values for features from testing data
    y_test : numpy array or pandas Series
        values for target from testing data
    save_path : str
        relative path of where the generated figures should be saved

    Returns
    ----------
        Returns multi-class confusion matrix (ndarray).
    """
    
    labels = np.unique(y_test)
    
    y_pred = model.predict(X_test)
    confusion_matrix = multilabel_confusion_matrix(y_test, y_pred, labels = labels)
    
    # Iterate over each label's confusion matrix
    # With reference to: sklearn.metrics.multilabel_confusion_matrix. In Scikit-learn documentation. 
    # https://scikit-learn.org/dev/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html

    for i in range(len(labels)):
        matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[i], 
                                        display_labels=["Not " + str(labels[i]), labels[i]])
        matrix.plot(cmap='Greens')
        plt.savefig(f"{save_path}confusion_matrix_class_{labels[i]}.png")
        print(f"Saved confusion_matrix_class_{labels[i]}.png to {save_path}")

    return confusion_matrix