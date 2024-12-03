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


<!-- ## How to Run the Data Analysis
1. Clone this repository:  
   ```bash
   git clone git@github.com:UBC-MDS/wine_quality_predictor_group1.git

2. Create the environment. In the root of the repository run:
   ```bash 
   conda env create --file environment.yaml

3. Ensure all dependencies are installed (see below).

4. Open the analysis notebook or script, e.g., analysis.ipynb. -->




## Dependencies
Python and packages listed in `environment.yml` file. This has been used in the creation of `conda-linux-64.lock` file which is used in creation of the Docker container.

## License
This project is licensed under the terms described in the LICENSE.md file, under MIT License and Creative Commons License. 
