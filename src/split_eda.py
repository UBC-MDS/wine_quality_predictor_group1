# Author: Bryan Lee
# Date: 2024-12-07

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Train-test split
def run_TrainTestSplit(clean_data_path, train_test_path):

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
    except FileNotFoundError as f:
        print(f"FileNotFoundError: The file path for saving splits is incorrect. Error: {f}")
    except KeyError as k:
        print(f"KeyError: {k}")
    except ValueError as v:
        print(f"ValueError: Invalid data type encountered. Error: {v}")
    except TypeError as t:
        print(f"TypeError: Incorrect type provided for train-test split. Error: {t}")
    except PermissionError as p:
        print(f"PermissionError: Insufficient permissions to write to {train_test_path}. Error: {p}")
    except Exception as e:
        print(f"Unexpected error during train-test split: {e}")

# EDA Charts
def run_eda_charts(figures_path, tables_path, train_test_path):

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
    except FileNotFoundError as f:
        print(f"FileNotFoundError: Could not save target distribution plot at {figures_path}. Error: {f}")
    except ValueError as v:
        print(f"ValueError: Invalid value in distribution plot. Error: {v}")
    except PermissionError as p:
        print(f"PermissionError: Insufficient permissions to save the plot. Error: {p}")
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
    except FileNotFoundError as f:
        print(f"FileNotFoundError: Could not save correlation heatmap at {figures_path}. Error: {f}")
    except ValueError as v:
        print(f"ValueError: Issue with correlation matrix generation. Error: {v}")
    except PermissionError as p:
        print(f"PermissionError: Insufficient permissions to save the heatmap. Error: {p}")
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
    except FileNotFoundError as f:
        print(f"FileNotFoundError: Could not save feature distribution plot at {figures_path}. Error: {f}")
    except ValueError as v:
        print(f"ValueError: Issue with KDE plot. Error: {v}")
    except PermissionError as p:
        print(f"PermissionError: Insufficient permissions to save the feature distribution plot. Error: {p}")
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
    except FileNotFoundError as f:
        print(f"FileNotFoundError: Could not save feature pairplot at {figures_path}. Error: {f}")
    except ValueError as v:
        print(f"ValueError: Issue with pairplot. Error: {v}")
    except PermissionError as p:
        print(f"PermissionError: Insufficient permissions to save the pairplot. Error: {p}")
    except Exception as e:
        print(f"Unexpected error during feature pairplot: {e}")