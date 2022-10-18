"""Train the model from a pre-processed dataset.

Run from the repository root as:
python dynamic_risk_assessment/training.py

Additionally, specify the config file by using:
python dynamic_risk_assessment/training.py --config_path conf/config.json
"""
import pickle
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dynamic_risk_assessment.utils import load_config


def process_data(
    data: pd.DataFrame,
    feature_names: List[str],
    label_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Process raw data to a training dataset.

    Arguments
    ---------
    data:
        Full raw dataset.

    feature_names:
        List of dataframe columns that make up the model features.

    label_name:
        The column name that provides the regression label.

    Returns
    -------
    features:
        Processed features

    labels:
        Processed labels
    """

    features = data[feature_names].values
    labels = data[label_name].values

    return features, labels


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_kwargs: dict,
    out_path: str
    ) -> LogisticRegression:
    """Initialize the regression model, train it, and store to file.

    Arguments
    ---------
    x_train:
        Input features

    y_train:
        Training labels

    model_kwargs:
        Scikit-learn model parameters

    out_path:
        File to store the trained model pickle.

    Returns
    -------
    A trained scikit-learn model instance.
    """

    model = LogisticRegression(**model_kwargs)

    # Train the model.
    model.fit(x_train, y_train)

    # Save model to disk.
    with open(out_path, "wb") as file:
        pickle.dump(model, file)

    return model

def main(args):
    """Model training function."""

    cfg = load_config(args.config_path)
    data = pd.read_csv(cfg["paths"]["ingested_data"])

    x_train, y_train = process_data(
        data, cfg["data"]["features"], cfg["data"]["label"])

    train_model(
        x_train, y_train, cfg["model"]["params"], cfg["paths"]["trained_model"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Config file path",
        default="./conf/config.json")

    args = parser.parse_args()

    main(args)
