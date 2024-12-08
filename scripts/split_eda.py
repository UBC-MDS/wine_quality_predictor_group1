# Author: Bryan Lee
# Date: 2024-12-07

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import click
import os
import pickle


@click.command()
@click.option("--clean_data_path", type=str, help="Path to pull raw data for train_test_split.")
@click.option("--train_test_path", type=str, help="Path to store and access data splits.")
@click.option("--figures_path", type=str, help="Path to save figures generated.")
def main(clean_data_path, train_test_path, figures_path) :

    # try-except for data splitting
    try:
        # importing the data from previous path for clean data
        df = pd.read_csv(clean_data_path)

        # creating X and Y for train test split
        X = df.drop('quality', axis=1) 
        y = df['quality']

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train.to_csv(os.path.join(train_test_path, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(train_test_path, "y_train.csv"), index=False)
        X_test.to_csv(os.path.join(train_test_path, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(train_test_path, "y_test.csv"), index=False)
        print("Testing and training splits successfully saved.")
    except Exception as e:
        print(f"Error: {e}")

    #try-except for EDA
    try:
        #Ensuring path exists for saving figures:
        os.makedirs(figures_path, exist_ok=True)

        # Describe plot
        print(X_train.describe())

        # Target Distribution Plot
        plt.figure(figsize=(8,4))
        sns.countplot(x=y_train)
        plt.title(f"Figure 1 - Distribution of Target Class in the Data Set")
        plt.savefig(f"{figures_path}/target_distribution_plot.png", format="png", dpi=300)
        print("target_distribution_plot.png successfully saved.")

        # Correlation Heatmap
        plt.figure(figsize=(10, 8)) 
        correlation_matrix = X_train.corr(method='pearson')
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
                annot_kws={'size': 10, 'color': 'black'}, linewidths=0.6)
        plt.title(f"Figure 2 - Wine Quality Features Heatmap - Pearson Correlation")
        plt.savefig(f"{figures_path}/correlation_heatmap.png", format = "png", dpi=300)
        print("correlation_heatmap.png successfully saved.")

        # KDE Plot for Feature Distributions
        num_features = len(X_train.columns)
        fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 2*num_features))

        # loop to create feature distribution plots
        for i, column in enumerate(X_train.columns):
            sns.kdeplot(df[column], ax=axes[i], fill=True)
            axes[i].set_title(f"Figure {i+3}: KDE for {column}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"{figures_path}/feature_distributions.png", format = "png", dpi=300)
        print("feature_distributions.png successfully saved.")

        # Pairplot for all features
        feature_pairplot = sns.pairplot(X_train, kind = 'reg', diag_kind = 'hist')
        feature_pairplot.fig.suptitle('Figure 15: Regression Pairplot for All Features', size = 30)
        feature_pairplot.fig.subplots_adjust(top = 0.94)
        plt.savefig(f"{figures_path}/feature_pairplots.png", format = "png", dpi=300)
        print("feature_pairplots.png successfully saved.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

