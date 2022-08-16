"""
This module trains two Scikit learn models on a Kaggle dataset to predict
the rate of credit card customer churn. The two models are a random forest
classifier and a logistic regression model.

Author: JL
Date created: 2022-08-16
"""

import os
from typing import Union
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

BANK_DATA_PTH = "./data/bank_data.csv"

CATEGORICAL_VARS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

FEATURE_NAMES = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn'
]

ScikitModelType = Union[RandomForestClassifier, LogisticRegression]
sns.set()

@dataclass
class ModelConfig:
    """Specifies experiment parameters for a scikit learn model."""
    name: str
    model: ScikitModelType
    model_kwargs: dict
    model_save_pth: str = "./models/"
    result_save_pth: str = "./images/results/"
    cross_val_grid: dict = None
    num_cv: int = None

def import_data(pth):
    """Returns dataframe for the csv found at pth.

    Parameters
    ----------
    pth : str
        Path to a .csv file.

    Returns
    -------
    data : pd.DataFrame
        Raw dataset.
    """
    data = pd.read_csv(pth)
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return data

def perform_eda(data, image_pth="./images/eda/"):
    """Perform exploratory data analysis and save figures to image_pth.

    Parameters
    ----------
    data : pd.Dataframe
        Raw dataset.

    image_pth : std, default="./images/eda/"
        Folder path for storing EDA images.

    Notes
    -----
    Creates the folder image_pth if does not already exist.
    """

    plot_variables = ["Churn", "Customer_Age", "Marital_Status",
                      "Total_Trans_Ct", "heatmap_all"]

    print("Starting exploratory data analysis...")

    num_nans = data.isnull().sum().sum()
    print("Number of missing values:", num_nans)

    os.makedirs(image_pth, exist_ok=True)
    for var in plot_variables:
        plt.figure(figsize=(20,10))

        # Histograms.
        if var in ["Churn", "Customer_Age"]:
            data[var].hist()

        # Bar plots.
        elif var in ["Marital_Status"]:
            data[var].value_counts(normalize=True).plot(kind="bar")

        # Distplots.
        elif var in ["Total_Trans_Ct"]:
            sns.histplot(data[var], stat="density", kde=True)

        # Heatmap.
        elif var == "heatmap_all":
            sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)

        else:
            print(f"No plotting instructions specified for '{var}'. Passing.")
            continue

        plt.title(var)
        plt.savefig(image_pth + f"{var}.png")

    print(f"Finished EDA! Saved images to {image_pth}.")

def category_mean_encoder(data, category_lst, response="Churn"):
    """Replace categorical column values with the corresponding mean churn rate.

    For instance, for each gender, add a column for each customer with the
    churn rate corresponding to the customer's gender.

    Parameters
    ----------
    data : pd.DataFrame
        Raw input data.

    category_lst : list
        List of all categorical variables to include in the analysis.

    response : str, default="Churn"
        The variable we try to predict.

    Returns
    -------
    data : pd.DataFrame
        New dataframe with added columns for categorical encodings.
    """

    for cat in category_lst:
        feat_name = cat + "_" + response # e.g. "Gender_Churn"
        cat_mean = data.groupby(cat).mean()[response]
        data[feat_name] = data[cat].apply(lambda val, means=cat_mean: means[val])

    return data

def perform_feature_engineering(data, category_lst, feature_lst):
    """Turn an initial raw dataset into a clean training data.

    Parameters
    ----------
    data : pd.DataFrame
        Initial raw dataset.

    Returns
    -------
    x_train, x_test, y_train, y_test : float array-like
        Scikit learn train_test_split outputs.

    info : dict
        Additional information from the data cleaning process.
    """

    data = category_mean_encoder(data, category_lst)

    features = data[feature_lst]
    labels = data["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size= 0.3, random_state=42)

    return x_train, x_test, y_train, y_test

def classification_report_image(y_train, y_test, train_preds, test_preds, model_cfg):
    """Make a classification report of a trained model and store it as image.

    Parameters
    ----------
    y_train : float array-like
        Training response values

    y_test : float array-like
        Test response values

    train_preds : float array-like
        Predictions from a trained model.

    test_preds : float array-like
        Test predictions from a trained model.

    model_cfg : ModelConfig
        A model configuration file.
    """

    test_report = str(classification_report(y_test, test_preds))
    train_report = str(classification_report(y_train, train_preds))
    font_kwargs = dict(fontdict={'fontsize': 10}, fontproperties = 'monospace')

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, f"{model_cfg.name} train", **font_kwargs)
    plt.text(0.01, 0.05, test_report, **font_kwargs)

    plt.text(0.01, 0.6, f"{model_cfg.name} test", **font_kwargs)
    plt.text(0.01, 0.7, train_report, **font_kwargs)
    plt.axis('off')
    plt.savefig(model_cfg.result_save_pth + model_cfg.name + "_report.png")
    plt.close()

def feature_importance_plot(model, feat_cols, save_path):
    """Plot feature importance ranking.

    Parameters
    ----------
    model : ScikitModel
        A trained scikit learn model instance.

    feat_cols : list
        A list of column names, corresponding to the training data features.

    save_path : str
        A location where to save the image.
    """

    importances = model.feature_importances_
    num_cols = len(feat_cols)

    # Sort feature importances in descending order.
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances.
    names = [feat_cols[i] for i in indices]

    plt.clf()
    plt.figure(figsize=(20,5))

    plt.title("Feature Importance")
    plt.ylabel('Importance')

    plt.bar(range(num_cols), importances[indices])

    plt.xticks(range(num_cols), names, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(x_train, x_test, y_train, y_test, cfg):
    """Train a model defined in cfg on the feature-label data.

    Parameters
    ----------
    x_train, x_test, y_train, y_test : float array-like
        Scikit learn train_test_split outputs.

    cfg : ModelConfig
        Defines a Scikit learn model details.

    Notes
    -----
    Stores the best model at cfg.model_save_pth.
    """

    print(f"Training model {cfg.name}...")
    model = cfg.model(**cfg.model_kwargs)

    if cfg.cross_val_grid is not None:
        assert cfg.num_cv is not None, "Specify 'num_cv' for cross validation."
        model = GridSearchCV(model, param_grid=cfg.cross_val_grid, cv=cfg.num_cv)

    model.fit(x_train, y_train)

    if cfg.cross_val_grid is not None:
        model = model.best_estimator_

    # Evaluation metrics
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    classification_report_image(y_train, y_test, train_preds, test_preds, cfg)

    # Save model to file.
    if cfg.model_save_pth is not None:
        joblib.dump(model, cfg.model_save_pth + cfg.name + ".pkl")

    print("Finished!")
    print(f"Stored model to {cfg.model_save_pth}.")
    print(f"Stored results to {cfg.result_save_pth}")
    print(30 * "-")

def train_models(x_train, x_test, y_train, y_test, feature_names):
    """Run the main training loop.

    Parameters
    ----------
    x_train, x_test, y_train, y_test : float array-like
        Scikit learn train_test_split outputs.

    feature_names : list
        List of feature names, order corresponding to the columns in x_train.
    """

    # MODEL CONFIGURATIONS
    # --------------------

    # Random forest classifier:
    cross_val_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']}

    rf_config = ModelConfig(
        name="random_forest",
        model=RandomForestClassifier,
        model_kwargs={"random_state": 42},
        cross_val_grid=cross_val_grid,
        num_cv=5
    )

    # Logistic regression:
    lr_config = ModelConfig(
        name="logistic_regression",
        model=LogisticRegression,
        model_kwargs={"solver": "lbfgs", "max_iter": 3000}
    )

    for cfg in [rf_config, lr_config]:
        train_model(x_train, x_test, y_train, y_test, cfg)

    # MODEL ANALYSIS
    # ---------------

    # Load the best models.
    rfc = joblib.load("./models/random_forest.pkl")
    lrc = joblib.load("./models/logistic_regression.pkl")

    feature_importance_plot(
        rfc, feature_names, "./images/results/forest_importance.png")

    # Make the ROC curve.
    fig, axis = plt.subplots()
    for model in [rfc, lrc]:
        plot_roc_curve(model, x_test, y_test, ax=axis, alpha=0.8)
    fig.savefig("./images/results/roc_curves.png")


if __name__ == "__main__":

    # DATA IMPORT
    # -----------
    BANK_DATA = import_data(pth=BANK_DATA_PTH)
    perform_eda(BANK_DATA)

    # FEATURE ENGINEERING
    # -------------------
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        BANK_DATA, CATEGORICAL_VARS, FEATURE_NAMES)

    # MODEL TRAINING
    # --------------
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, FEATURE_NAMES)
