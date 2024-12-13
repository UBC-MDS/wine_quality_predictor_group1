# Author: Bryan Lee
# Date: 2024-12-07

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Train-test split
def run_TrainTestSplit(clean_data_path, train_test_path):
    """
    Splits a dataset into training and testing subsets and saves them as CSV files.

    This function reads a cleaned dataset from the specified file path, separates the features 
    (X) and the target variable (y), performs a train-test split, and saves the resulting 
    subsets (X_train, y_train, X_test, y_test) into the specified directory. The function 
    handles potential errors during file reading and data splitting.

    Parameters:
    ----------
    clean_data_path : str
        The file path to the cleaned dataset in CSV format. The dataset is expected to have 
        a 'quality' column as the target variable and the rest of the columns as features.

    train_test_path : str
        The directory path where the train-test split CSV files (X_train, y_train, X_test, 
        y_test) will be saved. If the directory doesn't exist, it will be created.

    Returns:
    -------
    None
        This function does not return any value but will print success or error messages.

    Raises:
    ------
    Exception:
        If any error occurs during the loading of the CSV, train-test splitting, or file 
        saving, an exception message will be printed.
    """
    try:
        # Load data
        df = pd.read_csv(clean_data_path)
    except Exception as e:
        print(f'issue with reading csv: {e}') 

    try:
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        os.makedirs(train_test_path, exist_ok=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train.to_csv(os.path.join(train_test_path, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(train_test_path, "y_train.csv"), index=False)
        X_test.to_csv(os.path.join(train_test_path, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(train_test_path, "y_test.csv"), index=False)
        print('Train-test split functioning properly')
    except Exception as e:
        print(f"Unexpected error during train-test split: {e}")

# EDA Charts
def run_eda_charts(figures_path, tables_path, train_test_path):
    """
    Generates and saves exploratory data analysis (EDA) charts and tables for the training dataset.

    This function performs several EDA tasks on the training data:
    1. It saves a descriptive statistics table of the features.
    2. It creates and saves a distribution plot for the target variable.
    3. It generates a correlation heatmap for the features.
    4. It creates Kernel Density Estimation (KDE) plots for each feature's distribution.
    5. It creates and saves a pairplot showing the relationships between all features.

    All generated plots are saved in the specified directory, and tables are stored in another directory.

    Parameters:
    ----------
    figures_path : str
        The directory path where all the EDA plots (e.g., target distribution plot, heatmap, etc.) will be saved.
        If the directory does not exist, it will be created.

    tables_path : str
        The directory path where the EDA tables (e.g., descriptive statistics table) will be saved.
        If the directory does not exist, it will be created.

    train_test_path : str
        The directory path where the training data files (X_train.csv and y_train.csv) are stored. These files are 
        used to generate the EDA charts and tables.

    Returns:
    -------
    None
        This function does not return any value but will print success or error messages as it processes each step.

    Raises:
    ------
    Exception:
        If any error occurs during reading the training data, creating charts, or saving the files, an exception message
        will be printed. This includes errors in creating directories, plotting, or saving figures and tables.
    """
    try:
        os.makedirs(figures_path, exist_ok=True)
        os.makedirs(tables_path, exist_ok=True)
        X_train = pd.read_csv(os.path.join(train_test_path, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(train_test_path, 'y_train.csv'))

        # Describe plot
        describe_df = X_train.describe()
        describe_df.to_csv(os.path.join(tables_path, "describe_table.csv"))
        print('Describe table saved.')
    except Exception as e:
        print(f"Describe plot error: {e}")
    
    try:
        # Target Distribution Plot
        plt.figure(figsize=(8, 4))
        sns.countplot(x=y_train.iloc[:, 0])
        plt.title("Distribution of Target Class in the Data Set")
        plt.savefig(os.path.join(figures_path, "target_distribution_plot.png"), format="png", dpi=300)
        print('Target distribution plot saved.')
    except Exception as e:
        print(f"Unexpected error during target distribution plot: {e}")

    try:
        # Correlation Heatmap
        plt.figure(figsize=(7, 5))
        correlation_matrix = X_train.corr(method='pearson')
        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, annot_kws={'size': 10, 'color': 'black'}, linewidths=0.6)
        plt.title("Wine Quality Features Heatmap - Pearson Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, "correlation_heatmap.png"), format="png", dpi=300)
        print('Correlation heatmap saved.')
    except Exception as e:
        print(f"Unexpected error during correlation heatmap: {e}")

    try:
        # KDE Plot for Feature Distributions
        num_features = len(X_train.columns)
        fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 2*num_features))
        for i, column in enumerate(X_train.columns):
            sns.kdeplot(X_train[column], ax=axes[i], fill=True)
            axes[i].set_title(f"KDE for {column}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, "feature_distributions.png"), format="png", dpi=300)
        print("Feature distribution plot saved.")
    except Exception as e:
        print(f"Unexpected error during feature distribution plot: {e}")

    try:
        # Pairplot for all features
        plt.figure(figsize=(8, 4))
        feature_pairplot = sns.pairplot(X_train, kind='reg', diag_kind='hist')
        feature_pairplot.fig.suptitle('Regression Pairplot for All Features', size=30)
        feature_pairplot.fig.subplots_adjust(top=0.94)
        plt.savefig(os.path.join(figures_path, "feature_pairplots.png"), format="png", dpi=300)
        print("Feature Pairplot saved.")
    except Exception as e:
        print(f"Unexpected error during feature pairplot: {e}")