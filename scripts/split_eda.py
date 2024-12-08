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
"""
The main function for reading csv from path, performing train test split to create our training and testing 
and creating our EDA visualizations.

This function performs the following steps:
1. Reads the cleaned data from the provided `clean_data_path` as a CSV file.
2. Splits the data into features (X) and target (y), then performs a train-test split.
3. Saves the train and test splits (X_train, y_train, X_test, y_test) as CSV files to the specified `train_test_path`.
4. Generates a descriptive statistics plot for the training set features (X_train) and saves it as an image in the specified `figures_path`.
5. Creates and saves a distribution plot of the target variable (`y_train`), as well as a heatmap showing correlations between features.
6. Generates and saves Kernel Density Estimate (KDE) plots for feature distributions, and creates a pairplot of all features.

Parameters:
clean_data_path (str): The file path for the cleaned data CSV file.
train_test_path (str): The directory path where the train-test split CSV files should be saved.
figures_path (str): The directory path where the generated figures (plots) should be saved.

Returns:
None: The function performs data processing, saves CSV files, and generates visualizations.

Error Handling:
The function includes error handling for each major step, printing a message if any operation fails.
"""
    # importing the data from previous path for clean data
    try:
        df = pd.read_csv(clean_data_path)
        print('read csv ok')
    except:
        print('read csv not working') 

    # creating X and Y for train test split
    try: 
        X = df.drop('quality', axis=1) 
        y = df['quality']

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train.to_csv(os.path.join(train_test_path, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(train_test_path, "y_train.csv"), index=False)
        X_test.to_csv(os.path.join(train_test_path, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(train_test_path, "y_test.csv"), index=False)
        print('Train test split ok')
    except:
        print('train test split not working')

    #Ensuring path exists for saving figures:
    try: 
        os.makedirs(figures_path, exist_ok=True)

        # Describe plot
        ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        tbl = table(ax, pd.DataFrame(X_train.describe()), loc='center', colWidths=[0.2]*len(df.columns))
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        plt.savefig(f"{figre_path}/X_train_describe.png", bbox_inches='tight', pad_inches=0.05)
        print('describe plot working')

    except:
        print('describe plot not working')

    try:
        # Target Distribution Plot
        plt.figure(figsize=(8,4))
        sns.countplot(x=y_train)
        plt.title(f"Figure 1 - Distribution of Target Class in the Data Set")
        plt.savefig(f"{figures_path}/target_distribution_plot.png", format="png", dpi=300)

        # Correlation Heatmap
        plt.figure(figsize=(8, 6)) 
        correlation_matrix = X_train.corr(method='pearson')
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
                annot_kws={'size': 10, 'color': 'black'}, linewidths=0.6)
        plt.title(f"Figure 2 - Wine Quality Features Heatmap - Pearson Correlation")
        plt.savefig(f"{figures_path}/correlation_heatmap.png", format = "png", dpi=300)
        print('target distrubtion plot and correlation heatmap OK')
    
    except:
        print('target distribution plot and correlation heatmap not working')


    # KDE Plot for Feature Distributions
    try:
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

        # Pairplot for all features
        plt.figure(figsize=(8,4))
        feature_pairplot = sns.pairplot(X_train, kind = 'reg', diag_kind = 'hist')
        feature_pairplot.fig.suptitle('Figure 15: Regression Pairplot for All Features', size = 30)
        feature_pairplot.fig.subplots_adjust(top = 0.94)
        plt.savefig(f"{figures_path}/feature_pairplots.png", format = "png", dpi=300)
        print('feature distribution and feature pairplot ok')

    except: 
        print('figure distribution and pairplot not functioning.')

if __name__ == '__main__':
    main()
