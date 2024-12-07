import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import click
import pickle


@click.command()
@click.option("--train_test_path", type=str, help="Path to store and access data splits.")
@click.option("--cleaned_data_path", type=str, help="Path to pull raw data for train_test_split.")

def main(train_test_path, cleaned_data_path) :

    # importing the data from previous path for
    df = pd.read_csv(cleaned_data_path)

    # creating X and Y for train test split
    X = df.drop('quality', axis=1) 
    y = df['quality']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    write_csv(X_train, train_test_path,"X_train.csv")
    write_csv(X_test, train_test_path,"X_test.csv")
    write_csv(y_train, train_test_path,"y_train.csv")
    write_csv(y_test, train_test_path,"y_test.csv")

    # describe plot
    X_train.describe().T.style.background_gradient(cmap='Blues')

    # first distribution plot
    plt.figure(figsize=(8,4))
    sns.countplot(x=y_train)
    plt.title(f"Figure 1 - Distribution of Target Class in the Data Set")
    plt.show()

    plt.savefig("target_distribution_plot.png", format="png", dpi=300)

    # Correlation heatmap
    plt.figure(figsize=(10, 8)) 
    correlation_matrix = X_train.corr(method='pearson')
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
            annot_kws={'size': 10, 'color': 'black'}, linewidths=0.6)
    plt.title(f"Figure 2 - Wine Quality Features Heatmap - Pearson Correlation")

    plt.savefig("correlation_heatmap.png", format = "png", dpi=300)

    num_features = len(X_train.columns)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 2*num_features))

    # loop to create feature distribution plots
    for i, column in enumerate(X_train.columns):
        sns.kdeplot(df[column], ax=axes[i], fill=True)
        axes[i].set_title(f"Figure {i+3}: KDE for {column}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig("feature_distributions.png", format = "png", dpi=300)

    # pairplot for all features
    feature_pairplot = sns.pairplot(X_train, kind = 'reg', diag_kind = 'hist')
    feature_pairplot.fig.suptitle('Figure 15: Regression Pairplot for All Features', size = 30)
    feature_pairplot.fig.subplots_adjust(top = 0.94)

    plt.savefig("feature_pairplots.png", format = "png", dpi=300)


if __name__ == '__main__':
    main()

