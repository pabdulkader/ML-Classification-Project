# ML-Classification-Project
# Email Phishing Detection Project


## Project Description

This repository hosts a machine learning project dedicated to building a classification model for **detecting phishing emails**. The core of the project is the **Jupyter Notebook** (`EmailPhishingProject.ipynb`), which processes a comprehensive dataset of email features to train and evaluate a **Random Forest Classifier**.

The goal is to accurately classify incoming emails as either legitimate (Ham) or malicious (Phishing) based on quantifiable email characteristics.


## Files in this Repository

| File Name | Description |
| :--- | :--- |
| **EmailPhishingProject.ipynb** | The main Jupyter Notebook containing the data loading, preprocessing, model training, and evaluation for phishing detection. |
| **email\_phishing dataset.csv** | The dataset used for training and testing the model. It includes features like `num_words`, `num_links`, `num_email_addresses`, `num_spelling_errors`, and the target variable, `label` (0 for Ham, 1 for Phishing). |
| **README.md** | This file, providing an overview and instructions for the project. |


## Methodology and Model

The `EmailPhishingProject.ipynb` notebook implements the following machine learning pipeline:

1.  **Data Loading:** The **CSV dataset** is loaded using pandas.
2.  **Preprocessing:**
    * **Handling Missing Values:** The notebook addresses any null or missing data points.
    * **Feature Scaling:** Numerical features are standardized using **StandardScaler** to ensure all features contribute equally to the model training process.
3.  **Model Training:**
    * The data is split into training and testing sets (e.g., 70% train, 30% test).
    * A **Random Forest Classifier** (`RandomForestClassifier` from `sklearn.ensemble`) is trained on the preprocessed training data.
4.  **Evaluation:**
    * The model's performance is tested on the held-out set.
    * Key metrics are calculated, including **Accuracy Score** and the **Confusion Matrix**, to assess the classifier's ability to correctly identify phishing emails.


## Prerequisites

To run the analysis and train the model, you need a Python environment with the following libraries installed:

* **pandas**
* **numpy**
* **scikit-learn** (`sklearn`)
    * `RandomForestClassifier`
    * `train_test_split`
    * `StandardScaler`
    * `accuracy_score`, `confusion_matrix`

### Installation

You can install the necessary packages using `pip`:

```bash
pip install pandas numpy scikit-learn
