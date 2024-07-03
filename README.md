# Machine Learning Portfolio

This repository contains various machine learning projects completed as part of the CSCI316 course. Each project folder includes detailed implementations and analyses using different machine learning techniques and tools.

## Table of Contents
1. [Data Preprocessing and Student Performance Analysis](#data-preprocessing-and-student-performance-analysis)
2. [Iris and Mushroom Classification](#iris-and-mushroom-classification)
3. [Credit Score Prediction](#credit-score-prediction)
4. [Credit Scoring Analysis with PySpark](#credit-scoring-analysis-with-pyspark)

## Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn**
- **Scikit-Learn**
- **SQLAlchemy**
- **Jupyter Notebook**
- **TensorFlow**
- **PySpark**

## Project Details

### Data Preprocessing and Student Performance Analysis
#### [Data Preprocessing and Analysis](./A1/LimYeonjae_task1.ipynb)
- **Description**: This project involves the preprocessing and analysis of a dataset containing patient health information. Key tasks include identifying and handling missing values, performing z-score normalization, binning, applying one-hot encoding, and creating new attributes based on existing ones.
- **Technologies Used**: Python, Pandas, NumPy, Matplotlib, Seaborn
- **Implementation**: [Link to Notebook](./A1/LimYeonjae_task1.ipynb)
- **Highlights**:
  - Data cleaning and preprocessing
  - Exploratory Data Analysis (EDA)
  - Z-score normalization, binning, and one-hot encoding
#### [Student Performance Analysis](./A1/LimYeonjae_task2.ipynb)
- **Description**: This project involves the analysis of student performance data to predict outcomes such as graduation, enrollment, or dropout. Key tasks include data preprocessing, binning, feature selection, and model training using a decision tree classifier.
- **Technologies Used**: Python, Pandas, NumPy, Matplotlib, Scikit-Learn
- **Implementation**: [Link to Notebook](./A1/LimYeonjae_task2.ipynb)
- **Highlights**:
  - Data preprocessing and binning
  - Feature selection using chi-square tests
  - Model training and evaluation using a decision tree classifier with different criteria (entropy, gain ratio, gini impurity)
  - Achieved accuracy using different criteria and discussed pre-pruning techniques

### Iris and Mushroom Classification
#### [Iris Classification](./A2/LimYeonjae_task1.ipynb)
- **Description**: This project involves the analysis and classification of the Iris dataset using machine learning techniques. The goal is to classify the Iris species based on their features. Key tasks include data preprocessing, stratified sampling, and model training and evaluation using a Naive Bayes classifier.
- **Technologies Used**: Python, Pandas, NumPy
- **Implementation**: [Link to Notebook](./A2/LimYeonjae_task1.ipynb)
- **Highlights**:
  - Data preprocessing and stratified sampling
  - Model training and evaluation using a Naive Bayes classifier
  - Achieved high accuracy on both training and test sets
#### [Mushroom Classification](./A2/LimYeonjae_task2.ipynb)
- **Description**: This project involves the analysis and classification of a dataset using machine learning techniques. The dataset contains various attributes of mushrooms, and the goal is to predict whether a mushroom is edible or poisonous. Key tasks include data preprocessing, feature engineering, handling missing values, one-hot encoding, normalization, and model training and evaluation using logistic regression and artificial neural networks (ANN).
- **Technologies Used**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, TensorFlow
- **Implementation**: [Link to Notebook](./A2/LimYeonjae_task2.ipynb)
- **Highlights**:
  - Data transformation and feature engineering
  - Machine learning model training and evaluation using Scikit-Learn
  - Training and fine-tuning artificial neural networks using TensorFlow

### [Credit Score Prediction](./CSCI316_GA1.ipynb)
- **Description**: This project involves the analysis and prediction of credit scores using a dataset of customer financial information. The project includes data extraction, transformation, and loading (ETL) processes, as well as exploratory data analysis (EDA), data preprocessing, and machine learning model training.
- **Technologies Used**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, SQLAlchemy
- **Implementation**: [Link to Notebook](./CSCI316_GA1.ipynb)
- **Highlights**:
  - Comprehensive ETL process
  - Handling missing values and data cleaning
  - Feature engineering and machine learning model training
  - Model evaluation and performance metrics

### [Credit Scoring Analysis with PySpark](./CSCI316_GA1.ipynb)
- **Description**: This project focuses on credit scoring analysis using PySpark. It involves data extraction, transformation, and loading (ETL), along with comprehensive preprocessing, feature engineering, and model training using PySpark's MLlib.
- **Technologies Used**: Python, Pandas, NumPy, Matplotlib, Seaborn, PySpark, Spark MLlib
- **Implementation**: [Link to Notebook](./CSCI316_GA2.ipynb)
- **Highlights**:
  - Data extraction and cleaning using PySpark
  - Feature engineering and transformation
  - Machine learning model training and evaluation using Spark MLlib

