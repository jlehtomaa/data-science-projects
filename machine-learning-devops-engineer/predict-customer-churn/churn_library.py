"""
This module trains two Scikit learn models on a Kaggle dataset to predict
the rate of credit card customer churn. The two models are a random forest
classifier and a logistic regression model.

Author: JL
Date created: 2022-08-16
"""

import os

import hydra
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
sns.set()


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

def perform_eda(data, image_pth, plot_vars):
    """Perform exploratory data analysis and save figures to image_pth.

    Parameters
    ----------
    data : pd.Dataframe
        Raw dataset.

    image_pth : str
        Folder path for storing EDA images.

    plot_vars : lst
        List of feature names to plot during EDA.

    Notes
    -----
    Creates the folder image_pth if does not already exist.
    """

    num_nans = data.isnull().sum().sum()
    print("Number of missing values:", num_nans)

    os.makedirs(image_pth, exist_ok=True)
    for var in plot_vars:
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

    category_lst : list
        List of categorical features.

    feature_lst : list
        All feature column names to keep in the analysis.

    Returns
    -------
    train_data, test_data : float array-like
        Scikit learn train_test_split outputs.
    """

    data = category_mean_encoder(data, category_lst)

    features = data[feature_lst]
    labels = data["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size= 0.3, random_state=42)

    return [x_train, y_train], [x_test, y_test]

def make_classification_report(model, test_data, train_data, name, pth):
    """Make a classification report of a trained model and store it as image.

    Parameters
    ----------
    model : Scikit learn model instance
        A trained model.

    test_data : tuple float array-like
        Contains a (x_test, y_test) tuple.

    train_data : tuple float array-like
        Contains a (x_train, y_train) tuple.

    name : str
        Name of the model instance.

    pth : str
        A location for storing the classification report.
    """

    x_test, y_test = test_data
    test_preds = model.predict(x_test)

    x_train, y_train = train_data
    train_preds = model.predict(x_train)

    test_report = str(classification_report(y_test, test_preds))
    train_report = str(classification_report(y_train, train_preds))
    font_kwargs = dict(fontdict={'fontsize': 10}, fontproperties = 'monospace')

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, f"{name} train", **font_kwargs)
    plt.text(0.01, 0.05, test_report, **font_kwargs)

    plt.text(0.01, 0.6, f"{name} test", **font_kwargs)
    plt.text(0.01, 0.7, train_report, **font_kwargs)
    plt.axis('off')
    plt.savefig(pth + name + "_report.png")
    plt.close()

def train_model(name, cfg, train_data):
    """Train a Scikit learn model based on a config file.

    Parameters
    ----------

    name : str
        A name of the model instance.

    cfg : dict
        Model configuration details.

    train_data : tuple
        Training dataset tuple of the form (x_train, y_train).
    """

    if name == "random_forest":
        model_cls = RandomForestClassifier
    elif name == "logistic_regression":
        model_cls = LogisticRegression
    else:
        raise ValueError("Unknown Skicit model class.")

    model = model_cls(**cfg["params"])

    param_grid = cfg.get("cv_param_grid")
    if param_grid is not None:
        model = GridSearchCV(model, param_grid=dict(param_grid), cv=cfg["num_cv"])

    x_train, y_train = train_data
    model.fit(x_train, y_train)

    return model if param_grid is None else model.best_estimator_

def plot_roc_curves(names, x_test, y_test, model_pth, fig_pth):
    """Plot the receiver operating characteristic curves for all models.

    Parameters
    ----------

    names : str lst
        List of trained model names.

    x_test : float array-like
        Test data features

    y_test : float array-like
        Test data labels

    model_pth : str
        Folder to load trained models from.

    fig_pth : str
        Folder to store ROC curves in.
    """

    fig, axis = plt.subplots()
    for name in names:
        model = joblib.load(model_pth + name + ".pkl")
        plot_roc_curve(model, x_test, y_test, ax=axis, name=name)
    fig.savefig(fig_pth + "roc_curves.png")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """Run the full experiment based on a hydra config file specifications."""

    # DATA IMPORT AND EXPLORATORY ANALYSIS
    # ------------------------------------
    paths = cfg["paths"]
    data = import_data(paths["raw_data"])
    print("Starting exploratory data analysis...")
    perform_eda(data, paths["eda"], cfg["data_processing"]["plot_vars"])

    # FEATURE ENGINEERING
    # -------------------
    preproc = cfg["data_processing"]
    train_data, test_data = perform_feature_engineering(
        data, preproc["category_lst"], preproc["feature_lst"])

    # MODEL TRAININIG
    # ---------------
    for name in cfg["models"]:
        print(f"Training model {name}...")
        model = train_model(name, cfg["models"][name], train_data)

        make_classification_report(
            model, test_data, train_data, name, paths["results"])

        # Save best model on file.
        joblib.dump(model, paths["models"] + name + ".pkl")

    # ANALYSE MODELS
    # --------------
    models = list(cfg["models"].keys())
    plot_roc_curves(models, *test_data, paths["models"], paths["results"])

    print("All models trained successfully!")


if __name__ == "__main__":
    main()
