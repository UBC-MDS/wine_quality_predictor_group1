# Red Wine Quality Prediction

## Authors
- Yixuan Gao  
- Bryan Lee  
- Wangkai Zhu  
- Timothy Singh  

## Summary
The aim of this project is to build a classification model to predict the quality of red wine based on its physiochemical properties. The project addresses a multi-class classification problem where the target variable, wine quality, is an integer ranging from 0 (poor quality) to 10 (high quality).  

Several models were evaluated, including:
- **K-Nearest Neighbors (KNN)**  
- **Support Vector Machine with Radial Basis Function kernel (SVM RBF)**
- **Naive Bayes**
- **Logistic Regression**  
- **Decision Tree**

The methodology included hyperparameter tuning and 5-fold cross-validation to ensure optimal model performance. The best-performing model was determined based on accuracy and other relevant metrics.  

The best-performing model was the RBF SVM, which achieved a validation score of approximately 66% and a test set accuracy of around 58%. While the model demonstrated reasonable competence in predicting wines with mediocre quality ratings (5 or 6), its performance declined significantly for wines of higher or lower quality. The confusion matrices suggest challenges in differentiating certain classes, with class imbalances likely impacting performance (e.g., classes 3 and 8 appear to have many true negatives but no true positives). This indicates that the model struggles to handle outliers and extreme cases effectively.

### Dataset
The dataset used for this project is the **Red Wine Quality Dataset** from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). It consists of 1,599 observations with 11 continuous features, such as fixed acidity, volatile acidity, citric acid, and alcohol content.  

The dataset is referenced from the work of Paulo Cortez et al. ([details here](http://www3.dsi.uminho.pt/pcortez/wine/)).

---
## Running the Analysis file with the Docker Container
1. Clone repository - (https://github.com/UBC-MDS/wine_quality_predictor_group1).
2. Open Docker Hub and ensure the Docker Hub application is open and logged in with the correct credentials.
3. In your terminal use `cd wine_quality_predictor_group1` to switch to the newly created repository.
4. When with your terminal is in the new repository, run the command `docker-compose up`. 
    - The first time you will need to pull the image which may take a few minutes to load. 
5. Once the image loads, on your terminal click the link which starts with http://127.0.0.1, it will contain your token information for the Docker Container
    - Make sure you have no other instances of Jupyter Lab is opened on port 8888, as clicking this link will open a Jupyter Lab on this port.
6. You are able to access the files from this project, to locate the working project file 'notebooks/wine_predictor_analysis_report', and run cells as needed. Outputs, such as model performance metrics and visualizations, will be displayed or saved to the output/ directory.
7. After closing the container, run the command `docker-compose rm` to clean up the container.


## Scripts
### 1. Download Data
This script downloads or reads data stored in a `.zip` file and saves it locally.
```bash
python scripts/download_data.py --url=https://archive.ics.uci.edu/static/public/186/wine+quality.zip --write_to=data/raw/
```
- <url>: URL from internet to download `.zip` file (E.g. https://archive.ics.uci.edu/static/public/186/wine+quality.zip).
- <write_to>: Path to save the downloaded data (E.g. data/raw).


### 2. Clean Data
This script cleans the dataset by removing duplicates and handling missing values.
```bash
python scripts/clean_data.py --input_path=data/raw/raw_data.csv --output_path=data/processed/cleaned_data.csv --log_path=results/tables/
```
- <input_path>: Path to the raw data file (E.g. data/raw/raw_data.csv).
- <output_path>: Path to save the cleaned data (E.g. data/processed/cleaned_data.csv).
- <log-path>: Path to saves results/logs of data cleaning.


### 3. Data Validation
This script validates the data against the predefined schema.
```bash
python scripts/data_validation_script.py data/processed/cleaned_data.csv
```
- <input_path>: Path to the cleaned data (E.g. data/processed/cleaned_data.csv).


### 4. Data Splitting and Exploratory Data Analysis (EDA)
This script gets the cleaned data and applies train-test split.
4 csv files are created in a new `train_test_path`:
 - **X_train.csv**
 - **X_test.csv**
 - **y_train.csv**
 - **y_test.csv**

The EDA plots are saved as individual `.png` files. Charts should appear in the order below:
* `target_distribution_plot.png`
* `correlation_heatmap.png`
* `feature_distributions.png`
* `feature_pairplots.png`
```bash
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
```

- <clean_data_path>: Path to the cleaned data (E.g. clean_data_path=data/processed/cleaned_data.csv)
- <train_test_path>: Path to save the train-test splits of the data set. (E.g. data/processed/)
- <figures_path>: Path to save the figures generated from EDA. (E.g. results/figures/)
- <tables_path>: Path to save the tables generated from EDA. (E.g. results/tables/)

### 5. Preprocessing and Model Selection
This script creates a preprocessor, and performs 5-fold cross validation on different models. 
The scores from this cross-valiation are saved, as well as the model with the best evaluation score.
```bash
python scripts/preprocess_model_selection.py --train_data_path=data/processed/ --scores_path=results/tables/ --preprocessor_path=results/models/ --model_path=results/models/
```
- <train_data_path>: Relative path to retrieve training data.
- <scores_path>: Relative path to save training and validation scores.
- <preprocessor_path>: Relative path to save the preprocessor as .pickle file.
- <model_path>: Relative path to save best performing model as .pickle file.

### 6. Model Tuning
This script takes an SVC pipeline and tunes the model with RandomSearchCV.
```bash
python scripts/tuning_script.py results/models/base_model.pickle results/models/best_model.pickle data/processed/X_train.csv data/processed/y_train.csv data/processed/X_test.csv data/processed/y_test.csv
```
- <model_path>: Path to the retrieve pre-trained model file (.pickle).
- <best_model_path>: Path to save the fine-tuned model (.picke).
- <X_train_path>: Path to the training features (.CSV).
- <y_train_path>: Path to the training labels (.CSV).
- <X_test_path>: Path to the testing features (.CSV).
- <y_test_path>: Path to the testing labels (.CSV).

### 7. Model Evaluation
This script finds the accuracy of the model for predictions on the testing set.
It also creates and saves confusion matrices using the One vs Rest method of scoring.
```bash
python scripts/model_evaluation.py --tuned_model_path=results/models/best_model.pickle --test_split_path=data/processed/ --test_accuracy_path=results/tables/ --figures_path=results/figures/
```
- <tuned_model_path>: Relative path to the tuned model after hyperparameter tuning.
- <test_split_path>: Relative path to testing split of the data set.
- <test_accuracy_path>: Relative path to save test accuracy.
- <figures_path>: Path to save any figures from evaluation.

### 8. Report Generation


## Dependencies
Python and packages listed in `environment.yml` file. This has been used in the creation of `conda-linux-64.lock` file which is used in creation of the Docker container.

## License
This project is licensed under the terms described in the LICENSE.md file, under MIT License and Creative Commons License. 
