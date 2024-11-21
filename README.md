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
- **Logistic Regression**  
- **Decision Tree**

The methodology included hyperparameter tuning and 5-fold cross-validation to ensure optimal model performance. The best-performing model was determined based on accuracy and other relevant metrics.  

### Dataset
The dataset used for this project is the **Red Wine Quality Dataset** from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). It consists of 1,599 observations with 11 continuous features, such as fixed acidity, volatile acidity, citric acid, and alcohol content.  

The dataset is referenced from the work of Paulo Cortez et al. ([details here](http://www3.dsi.uminho.pt/pcortez/wine/)).

---

## How to Run the Data Analysis
1. Clone this repository:  
   ```bash
   git clone git@github.com:UBC-MDS/wine_quality_predictor_group1.git

2. Create the environment. In the root of the repository run:
   ```bash 
   conda env create --file environment.yaml

3. Ensure all dependencies are installed (see below).

4. Open the analysis notebook or script, e.g., analysis.ipynb.


Outputs, such as model performance metrics and visualizations, will be displayed or saved to the output/ directory.

## Dependencies
Python and packages listed in environment.yaml file.

## License
This project is licensed under the terms described in the LICENSE.md file.