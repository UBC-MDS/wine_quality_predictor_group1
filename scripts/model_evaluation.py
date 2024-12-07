import pandas as pd
import numpy as np
import click
import pickle

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay


@click.command()
@click.option("--tuned_model_path", type=str, help="Path to access tuned model.")
@click.option("--preprocessor_path", type=str, help="Path to access preprocessor.")
@click.option("--test_accuracy_path", type=str, help="Path to save the test accuracy score.")
@click.option("--figures_path", type=str, help="Path to save any figures from results.")
def main(tuned_model_path, preprocessor_path, test_accuracy_path, figures_path):
    with open(tuned_model_path + "tuned_model.pickle", 'rb') as f:
        best_model = pickle.load(f)
        
    test_score = best_model.score(X_test, y_test)
    test_accuracy_df = pd.DataFrame({"accuracy": [test_score]})

    # Save test_accuracy_df to .csv file to scores_path
    write_csv(test_accuracy_df, test_accuracy_path, "test_accuracy.csv")

    # Multi-class Confusion Matrix
    labels = np.unique(y_test)
    
    y_pred = best_model.predict(X_test)
    confusion_matrix = multilabel_confusion_matrix(y_test, y_pred, labels = labels)
    
    # Iterate over each label's confusion matrix
    # With reference to: sklearn.metrics.multilabel_confusion_matrix. In Scikit-learn documentation. 
    # Retrieved November 23, 2024, from https://scikit-learn.org/dev/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
    
    for i in range(len(labels)):
        matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[i], 
                                        display_labels=["Not " + str(labels[i]), labels[i]])
        matrix.plot(cmap='Greens')
        plt.save(f"{figures_path}_class_{labels[i]}.png")
        
if __name__ ==  "__main__":
    main()
    
            